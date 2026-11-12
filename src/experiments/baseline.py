"""E0: create (or verify) the common starting checkpoint for a seed and record val-only metrics."""
from __future__ import annotations

import json
import time
from typing import Any

import torch
from torch import nn

from src.data.dataset import get_loader
from src.models.alexnet import build_model, save_model_card, set_weights_dir, pretrained_weights_sha256
from src.models.checkpoint import (baseline_path, register_checkpoint, require_common_checkpoint,
                                   save_checkpoint, read_registry)
from src.training.evaluate import evaluate
from src.training.warmup import (splits_manifest_hash, assemble_model, ensure_feature_cache,
                                 train_head, verify_agreement)
from src.utils.config import Config, config_hash, resolve_path
from src.utils.logger import JsonlLogger
from src.utils.reproducibility import enable_deterministic, set_seed, sha256_tensor_dict


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_common_model(cfg: Config, seed: int, device: torch.device,
                       log: JsonlLogger) -> tuple[nn.Module, dict[str, Any]]:
    """Warm up the head on cached features and return (assembled model, training info)."""
    t0 = time.perf_counter()
    cache = ensure_feature_cache(cfg, device)
    t_cache = time.perf_counter() - t0
    set_seed(seed)
    t0 = time.perf_counter()
    head_state, history = train_head(cache["train"], cache["val_edit"], cache["val_select"],
                                     cfg.e0, seed, device)
    t_train = time.perf_counter() - t0
    model = assemble_model(head_state, seed, cfg)
    agreement = verify_agreement(model, cache["val_select"], cfg, device)
    best = max(history, key=lambda h: (h["val_select_acc"], -h["epoch"]))
    log.log("warmup_done", best_epoch=best["epoch"], best_val_select_acc=best["val_select_acc"],
            agreement=agreement, cache_seconds=t_cache, train_seconds=t_train)
    return model, {"history": history, "best_epoch": best["epoch"], "agreement": agreement,
                   "cache_seconds": t_cache, "train_seconds": t_train}


def _val_metrics(model: nn.Module, cfg: Config, device: torch.device) -> dict[str, Any]:
    model = model.to(device)
    out = {}
    for split in ("val_edit", "val_select"):
        result = evaluate(model, get_loader(split, cfg), device, amp=cfg.training.eval_amp,
                          num_classes=cfg.data.num_classes)
        out[split] = result.to_dict()
    return out


def run_baseline(cfg: Config, seed: int, log: JsonlLogger, reproduce_check: bool = False) -> dict[str, Any]:
    """Create or verify the seed's common checkpoint, then write results/E0/seed{S}/metrics.json."""
    device = _device()
    set_weights_dir(cfg)
    if cfg.training.deterministic:
        enable_deterministic(cfg.training.warn_only_deterministic)
    save_model_card(build_model(cfg.data.num_classes, seed, pretrained=False), resolve_path(cfg.paths.model_card))
    path, info = baseline_path(cfg, seed), {}
    if str(seed) not in read_registry(cfg):
        model, info = train_common_model(cfg, seed, device, log)
        meta = {"seed": seed, "config_hash": config_hash(cfg), "split_manifest_sha256": splits_manifest_hash(cfg),
                "weights_sha256": pretrained_weights_sha256(), "warmup_history": info["history"],
                "best_epoch": info["best_epoch"], "agreement": info["agreement"], "e0": cfg.e0.model_dump()}
        state_hash = save_checkpoint(model, path, meta)
        register_checkpoint(cfg, seed, path, state_hash)
        log.log("checkpoint_created", path=str(path), sha256=state_hash)
    model, state_hash = require_common_checkpoint(cfg, seed, device)
    log.log("checkpoint_verified", sha256=state_hash)
    if reproduce_check:
        again, _ = train_common_model(cfg, seed, device, log)
        again_hash = sha256_tensor_dict(again.state_dict())
        print(f"reproduce check seed {seed}: registered={state_hash} retrained={again_hash} "
              f"{'MATCH' if again_hash == state_hash else 'MISMATCH'}")
        log.log("reproduce_check", registered=state_hash, retrained=again_hash, match=again_hash == state_hash)
    metrics = {"experiment": "E0", "seed": seed, "config_hash": config_hash(cfg),
               "checkpoint_sha256": state_hash, **_val_metrics(model, cfg, device)}
    out = resolve_path(cfg.paths.e0_results_dir) / f"seed{seed}" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for split in ("val_edit", "val_select"):
        m = metrics[split]
        print(f"seed {seed} {split}: acc={m['accuracy']:.4f} top5={m['top5_accuracy']:.4f} "
              f"loss={m['loss']:.4f} f1={m['macro_f1']:.4f} {m['images_per_sec']:.0f} img/s ({m['seconds']:.1f}s)")
    return metrics
