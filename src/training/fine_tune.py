"""Full fine-tuning loop (E1): AdamW, warm-up + cosine, AMP, per-epoch val_select, resumable."""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.evaluate import evaluate
from src.utils.logger import JsonlLogger

RESUME_NAME, BEST_NAME = "resume.pt", "best_state.pt"
SELECT_SPLIT, EDIT_SPLIT = "val_select", "val_edit"


@dataclass(frozen=True)
class FitSettings:
    """Hyper-parameters for one fine-tuning run."""

    backbone_lr: float
    classifier_lr: float
    weight_decay: float
    epochs: int
    warmup_epochs: int
    accum_steps: int
    amp: bool
    seed: int
    num_classes: int = 10


@dataclass
class FitResult:
    """Outcome of ``fit``; ``best_state`` is the CPU state dict with best val_select accuracy."""

    history: list[dict[str, float]]
    best_epoch: int
    best_state: dict[str, torch.Tensor]
    best_metrics: dict[str, dict[str, Any]]
    compute: dict[str, float]
    accum_steps: int
    completed: bool


def build_optimizer(model: nn.Module, s: FitSettings) -> torch.optim.AdamW:
    """Two groups: ``features.*`` at backbone_lr, ``classifier.*`` at classifier_lr."""
    backbone = [p for n, p in model.named_parameters() if n.startswith("features.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("features.")]
    return torch.optim.AdamW([{"params": backbone, "lr": s.backbone_lr},
                              {"params": head, "lr": s.classifier_lr}], weight_decay=s.weight_decay)


def lr_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warm-up over ``warmup_steps`` then cosine decay to 0 at ``total_steps``."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_save(obj: Any, path: Path, attempts: int = 5, wait_seconds: float = 30.0) -> None:
    """Write to ``path.tmp`` then rename; retry on write errors (a momentarily full disk must not kill a run)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(attempts):
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)
            return
        except (RuntimeError, OSError):
            tmp.unlink(missing_ok=True)
            if attempt == attempts - 1:
                raise
            time.sleep(wait_seconds)


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _train_batch(model: nn.Module, opt: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
                 images: torch.Tensor, labels: torch.Tensor, accum: int, device: torch.device,
                 amp: bool) -> float:
    """One optimizer step on a batch, split into ``accum`` micro-batches (loss averaged over the batch)."""
    opt.zero_grad(set_to_none=True)
    total = 0.0
    for img, lab in zip(images.chunk(accum), labels.chunk(accum)):
        img, lab = img.to(device, non_blocking=True), lab.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            loss = nn.functional.cross_entropy(model(img).float(), lab, reduction="sum") / len(labels)
        scaler.scale(loss).backward()
        total += loss.item()
    scaler.step(opt)
    scaler.update()
    return total


def _train_epoch(model, loader, opt, sched, scaler, device, amp, accum, log, max_batches):
    """Train one epoch; on CUDA OOM double ``accum`` (logged) and retry the same batch."""
    model.train()
    loss_sum, n_images = 0.0, 0
    for b, (images, labels) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        while True:
            try:
                loss = _train_batch(model, opt, scaler, images, labels, accum, device, amp)
                break
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                accum *= 2
                log.log("oom_fallback", accum_steps=accum, batch=b)
                if accum > len(labels):
                    raise
        sched.step()
        loss_sum += loss * len(labels)
        n_images += len(labels)
    return loss_sum / max(1, n_images), n_images, accum


def fit(model: nn.Module, make_train_loader: Callable[[int], DataLoader],
        val_loaders: dict[str, DataLoader], s: FitSettings, device: torch.device, out_dir: Path,
        log: JsonlLogger, max_batches: int | None = None,
        stop_after_epoch: int | None = None) -> FitResult:
    """Fine-tune ``model`` (all layers trainable); resumes from ``out_dir/resume.pt`` if present.

    Selects the epoch with the best val_select accuracy (ties keep the earlier epoch). The test set is
    never touched. ``stop_after_epoch`` ends early (used to test resume).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    opt = build_optimizer(model, s)
    steps_per_epoch = len(make_train_loader(0)) if max_batches is None else max_batches
    warm, total = s.warmup_epochs * steps_per_epoch, s.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: lr_factor(step, warm, total))
    use_amp = s.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float]] = []
    best = {"acc": -1.0, "epoch": 0, "metrics": {}}
    compute = {"gpu_seconds": 0.0, "wall_seconds": 0.0, "images_seen": 0, "val_evaluations": 0, "epochs": 0}
    accum, start_epoch = s.accum_steps, 1
    resume_path, best_path = out_dir / RESUME_NAME, out_dir / BEST_NAME
    if resume_path.is_file():
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        _set_rng_state(ck["rng"])
        history, best, compute, accum = ck["history"], ck["best"], ck["compute"], ck["accum"]
        start_epoch = ck["epoch"] + 1
        log.log("resumed", epoch=ck["epoch"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, s.epochs + 1):
        if stop_after_epoch is not None and epoch > stop_after_epoch:
            break
        wall0 = time.perf_counter()
        ev0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        ev1 = torch.cuda.Event(enable_timing=True) if ev0 else None
        if ev0:
            ev0.record()
        train_loss, n_img, accum = _train_epoch(model, make_train_loader(epoch), opt, sched, scaler,
                                                device, use_amp, accum, log, max_batches)
        if ev1:
            ev1.record()
            torch.cuda.synchronize()
        train_seconds = time.perf_counter() - wall0
        gpu_seconds = ev0.elapsed_time(ev1) / 1000 if ev0 else 0.0
        ev = {name: evaluate(model, loader, device, amp=False, num_classes=s.num_classes)
              for name, loader in val_loaders.items()}
        compute["val_evaluations"] += len(ev)
        row = {"epoch": epoch, "train_loss": train_loss, "train_seconds": train_seconds,
               "gpu_seconds": gpu_seconds, "eval_seconds": sum(r.seconds for r in ev.values()),
               "lr_backbone": opt.param_groups[0]["lr"], "lr_classifier": opt.param_groups[1]["lr"]}
        for name, r in ev.items():
            row[f"{name}_acc"], row[f"{name}_loss"] = r.accuracy, r.loss
        history.append(row)
        compute["images_seen"] += n_img
        compute["gpu_seconds"] += gpu_seconds
        compute["wall_seconds"] += time.perf_counter() - wall0
        compute["epochs"] = epoch
        if ev[SELECT_SPLIT].accuracy > best["acc"]:
            best = {"acc": ev[SELECT_SPLIT].accuracy, "epoch": epoch,
                    "metrics": {n: r.to_dict() for n, r in ev.items()}}
            _atomic_save(_cpu_state(model), best_path)
        if device.type == "cuda":
            compute["peak_gpu_memory_mb"] = max(compute.get("peak_gpu_memory_mb", 0.0),
                                                torch.cuda.max_memory_allocated(device) / 2**20)
        log.log("epoch", **row, accum_steps=accum)
        _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                      "scaler": scaler.state_dict(), "rng": _rng_state(), "history": history, "best": best,
                      "compute": compute, "accum": accum, "epoch": epoch}, resume_path)
    done = bool(history) and history[-1]["epoch"] == s.epochs
    compute.setdefault("peak_gpu_memory_mb", 0.0)
    best_state = torch.load(best_path, map_location="cpu", weights_only=True) if best_path.is_file() else {}
    return FitResult(history, best["epoch"], best_state, best["metrics"], compute, accum, done)
