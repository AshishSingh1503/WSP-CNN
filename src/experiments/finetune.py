"""E1: full fine-tuning reference arm, plus the seed-0 learning-rate sweep (val_select only)."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

from src.data.dataset import get_loader
from src.models.checkpoint import clone_model, require_common_checkpoint, save_checkpoint
from src.training.fine_tune import FitResult, FitSettings, fit
from src.utils.config import Config, config_hash, resolve_path
from src.utils.logger import JsonlLogger
from src.utils.param_diff import diff_stats
from src.utils.reproducibility import enable_deterministic, set_seed

SWEEP_COLUMNS = ["backbone_lr", "classifier_lr", "epochs", "best_epoch", "best_val_select_acc",
                 "best_val_select_loss", "val_edit_acc_at_best", "final_val_select_acc",
                 "minutes", "peak_gpu_memory_mb"]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _settings(cfg: Config, seed: int, backbone_lr: float, classifier_lr: float, epochs: int) -> FitSettings:
    return FitSettings(backbone_lr=backbone_lr, classifier_lr=classifier_lr,
                       weight_decay=cfg.e1.weight_decay, epochs=epochs,
                       warmup_epochs=cfg.e1.warmup_epochs, accum_steps=cfg.e1.accum_steps,
                       amp=cfg.training.amp, seed=seed, num_classes=cfg.data.num_classes)


def _run_fit(cfg: Config, seed: int, s: FitSettings, out_dir: Path, log: JsonlLogger,
             ) -> tuple[nn.Module, str, FitResult]:
    """Start from the registered common checkpoint of ``seed`` and fine-tune every layer."""
    device = _device()
    model, start_hash = require_common_checkpoint(cfg, seed, device)
    if cfg.training.deterministic:
        enable_deterministic(cfg.training.warn_only_deterministic)
    set_seed(seed)
    reference = clone_model(model).cpu()
    result = fit(model, lambda epoch: get_loader("train", cfg, seed=seed * 1000 + epoch),
                 {"val_select": get_loader("val_select", cfg), "val_edit": get_loader("val_edit", cfg)},
                 s, device, out_dir, log)
    return reference, start_hash, result


def _e0_val_select(cfg: Config, seed: int) -> dict[str, Any] | None:
    path = resolve_path(cfg.paths.e0_results_dir) / f"seed{seed}" / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))["val_select"] if path.is_file() else None


def run_finetune(cfg: Config, seed: int, log: JsonlLogger) -> dict[str, Any]:
    """Full schedule for one seed with the configured learning rates; writes E1 results."""
    ckpt_dir = resolve_path(cfg.paths.finetune_dir) / f"seed{seed}"
    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        print(f"seed {seed}: E1 already complete ({best_path}); nothing to do")
        return {}
    s = _settings(cfg, seed, cfg.e1.backbone_lr, cfg.e1.classifier_lr, cfg.e1.epochs)
    reference, start_hash, result = _run_fit(cfg, seed, s, ckpt_dir, log)
    if not result.completed:
        raise RuntimeError("fit stopped before the final epoch")
    model = clone_model(reference)
    model.load_state_dict(result.best_state)
    diff = diff_stats(reference, model)
    meta = {"seed": seed, "config_hash": config_hash(cfg), "start_checkpoint_sha256": start_hash,
            "settings": s.__dict__, "best_epoch": result.best_epoch, "history": result.history,
            "compute": result.compute, "arm": "E1"}
    state_hash = save_checkpoint(model, best_path, meta)
    out_dir = resolve_path(cfg.paths.e1_results_dir) / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.history).to_csv(out_dir / "history.csv", index=False)
    e0 = _e0_val_select(cfg, seed)
    metrics = {"experiment": "E1", "seed": seed, "config_hash": config_hash(cfg),
               "start_checkpoint_sha256": start_hash, "checkpoint_sha256": state_hash,
               "settings": s.__dict__, "best_epoch": result.best_epoch, "accum_steps": result.accum_steps,
               "val_select": result.best_metrics["val_select"], "val_edit": result.best_metrics["val_edit"],
               "e0_val_select_accuracy": e0["accuracy"] if e0 else None,
               "compute": result.compute, "param_diff": diff}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for name in ("resume.pt", "best_state.pt"):
        (ckpt_dir / name).unlink(missing_ok=True)
    _print_seed_summary(seed, metrics, result)
    return metrics


def _print_seed_summary(seed: int, m: dict[str, Any], result: FitResult) -> None:
    c = m["compute"]
    per_epoch = c["wall_seconds"] / max(1, c["epochs"])
    e0 = m["e0_val_select_accuracy"]
    print(f"seed {seed}: val_select acc {m['val_select']['accuracy']:.4f} (best epoch {m['best_epoch']}/"
          f"{c['epochs']}), E0 {e0 if e0 is None else round(e0, 4)} | {per_epoch:.0f}s/epoch incl. eval | "
          f"GPU {c['gpu_seconds']:.0f}s wall {c['wall_seconds']:.0f}s | peak {c['peak_gpu_memory_mb']:.0f} MB | "
          f"changed {m['param_diff']['n_changed']:,}/{m['param_diff']['n_total']:,} "
          f"rel norm {m['param_diff']['global_rel_norm']:.4f} | accum {m['accum_steps']}")


def run_sweep(cfg: Config, log: JsonlLogger) -> None:
    """Seed-0 grid over (backbone_lr, classifier_lr), scored on val_select only; resumable per config."""
    out_csv = resolve_path(cfg.paths.e1_results_dir) / "tuning_sweep.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out_csv) if out_csv.is_file() else pd.DataFrame(columns=SWEEP_COLUMNS)
    for b in cfg.e1.sweep_backbone_lrs:
        for c in cfg.e1.sweep_classifier_lrs:
            if ((done["backbone_lr"] == b) & (done["classifier_lr"] == c)).any():
                continue
            s = _settings(cfg, cfg.e1.sweep_seed, b, c, cfg.e1.sweep_epochs)
            work = resolve_path(cfg.paths.finetune_dir) / "sweep" / f"b{b:g}_c{c:g}"
            _, _, r = _run_fit(cfg, cfg.e1.sweep_seed, s, work, log)
            best = r.best_metrics
            row = {"backbone_lr": b, "classifier_lr": c, "epochs": s.epochs, "best_epoch": r.best_epoch,
                   "best_val_select_acc": best["val_select"]["accuracy"],
                   "best_val_select_loss": best["val_select"]["loss"],
                   "val_edit_acc_at_best": best["val_edit"]["accuracy"],
                   "final_val_select_acc": r.history[-1]["val_select_acc"],
                   "minutes": r.compute["wall_seconds"] / 60, "peak_gpu_memory_mb": r.compute["peak_gpu_memory_mb"]}
            with out_csv.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=SWEEP_COLUMNS)
                if fh.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            shutil.rmtree(work, ignore_errors=True)
            done = pd.read_csv(out_csv)
            print(row)
    table = pd.read_csv(out_csv).sort_values(["best_val_select_acc", "backbone_lr", "classifier_lr"],
                                             ascending=[False, True, True])
    print(table.to_string(index=False))
    top = table.iloc[0]
    print(f"BEST (val_select, ties -> smaller lrs): backbone_lr={top.backbone_lr:g} classifier_lr={top.classifier_lr:g}")
