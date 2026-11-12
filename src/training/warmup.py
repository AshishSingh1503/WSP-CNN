"""Head-only warm-up on cached penultimate features (backbone frozen => features computed once)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.data.dataset import get_loader
from src.models.alexnet import (build_model, feature_extractor, init_head,
                                pretrained_weights_sha256)
from src.utils.config import Config, E0Config, resolve_path
from src.utils.hashing import sha256_file

Features = dict[str, torch.Tensor]  # {"features": [N, 4096] fp16, "labels": [N] int64}
SPLITS = ("train", "val_edit", "val_select")


def _manifest_path(cfg: Config) -> Path:
    return resolve_path(cfg.paths.features_dir) / "manifest.json"


def splits_manifest_hash(cfg: Config) -> str:
    return sha256_file(resolve_path(cfg.paths.splits_dir) / "manifest.json")


@torch.no_grad()
def compute_split_features(model: nn.Module, loader: Any, device: torch.device) -> Features:
    """fp32 eval-mode features (dropout off) cast to float16, in loader order."""
    extractor = feature_extractor(model).to(device).eval()
    feats, labels = [], []
    for images, y in loader:
        feats.append(extractor(images.to(device)).half().cpu())
        labels.append(y)
    return {"features": torch.cat(feats), "labels": torch.cat(labels).long()}


def ensure_feature_cache(cfg: Config, device: torch.device) -> dict[str, Features]:
    """Load cached features after hash verification, computing them first if absent.

    Features use the deterministic EVAL transform for every split (no augmentation).
    """
    out_dir = resolve_path(cfg.paths.features_dir)
    manifest_path = _manifest_path(cfg)
    if not manifest_path.exists():
        _build_cache(cfg, device, out_dir, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["splits_manifest_sha256"] != splits_manifest_hash(cfg):
        raise ValueError("feature cache was built from different split files; delete data/features")
    cache = {}
    for name in SPLITS:
        path = out_dir / manifest["files"][name]["file"]
        if sha256_file(path) != manifest["files"][name]["sha256"]:
            raise ValueError(f"feature cache hash mismatch: {path}")
        cache[name] = torch.load(path, map_location="cpu", weights_only=True)
    return cache


def _build_cache(cfg: Config, device: torch.device, out_dir: Path, manifest_path: Path) -> None:
    """Compute and save features for all three splits plus a hash manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg.data.num_classes, seed=0, weights_name=cfg.model.pretrained_weights)
    files: dict[str, Any] = {}
    for name in SPLITS:
        loader = get_loader(name, cfg, augment=False)
        data = compute_split_features(model, loader, device)
        path = out_dir / f"{name}.pt"
        torch.save(data, path)
        files[name] = {"file": path.name, "n": len(data["labels"]), "sha256": sha256_file(path)}
    manifest = {"dtype": cfg.e0.feature_dtype, "transform": "eval (no augmentation)",
                "weights_sha256": pretrained_weights_sha256(),
                "splits_manifest_sha256": splits_manifest_hash(cfg), "files": files}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _accuracy(head: nn.Linear, data: Features, device: torch.device) -> float:
    with torch.no_grad():
        logits = head(data["features"].to(device).float())
        return (logits.argmax(1).cpu() == data["labels"]).float().mean().item()


def train_head(train: Features, val_edit: Features, val_select: Features, cfg: E0Config,
               seed: int, device: torch.device) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    """Train ``Linear(4096, C)`` with AdamW + cosine; return (best-val_select head state, history)."""
    num_classes = int(train["labels"].max()) + 1
    head = nn.Linear(train["features"].shape[1], num_classes)
    init_head(head, seed)
    head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    x, y = train["features"].to(device), train["labels"].to(device)
    gen = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    best_acc, best_state = -1.0, None
    for epoch in range(1, cfg.epochs + 1):
        head.train()
        perm = torch.randperm(len(y), generator=gen).to(device)
        total = 0.0
        for i in range(0, len(y), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            loss = nn.functional.cross_entropy(head(x[idx].float()), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        lr = opt.param_groups[0]["lr"]
        sched.step()
        head.eval()
        acc_edit, acc_select = _accuracy(head, val_edit, device), _accuracy(head, val_select, device)
        history.append({"epoch": epoch, "train_loss": total / len(y), "val_edit_acc": acc_edit,
                        "val_select_acc": acc_select, "lr": lr})
        if acc_select > best_acc:  # ties keep the earlier epoch
            best_acc = acc_select
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    assert best_state is not None
    return best_state, history


def assemble_model(head_state: dict[str, torch.Tensor], seed: int, cfg: Config) -> nn.Module:
    """Pretrained AlexNet whose classifier[6] is the trained head."""
    model = build_model(cfg.data.num_classes, seed=seed, weights_name=cfg.model.pretrained_weights)
    model.classifier[6].load_state_dict(head_state)
    return model


@torch.no_grad()
def verify_agreement(model: nn.Module, val_select_cache: Features, cfg: Config,
                     device: torch.device) -> dict[str, float]:
    """Full-model fp32 forward vs head-on-cached-features on the first N val_select images.

    Raises if the prediction agreement is below ``cfg.e0.min_agreement``.
    """
    n = cfg.e0.agreement_images
    model = model.to(device).eval()
    loader = get_loader("val_select", cfg, batch_size=cfg.loader.batch_size_eval)
    full_logits, seen = [], 0
    for images, _ in loader:
        full_logits.append(model(images.to(device)).float().cpu())
        seen += len(images)
        if seen >= n:
            break
    full = torch.cat(full_logits)[:n]
    cached = model.classifier[6](val_select_cache["features"][:n].to(device).float()).cpu()
    agreement = (full.argmax(1) == cached.argmax(1)).float().mean().item()
    report = {"n": n, "agreement": agreement, "max_abs_logit_diff": (full - cached).abs().max().item()}
    if agreement < cfg.e0.min_agreement:
        raise RuntimeError(f"cached-feature path disagrees with full model: {report}")
    return report
