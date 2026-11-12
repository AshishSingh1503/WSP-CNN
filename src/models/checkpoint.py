"""Checkpoint save/load with state-dict hashing, plus the common-checkpoint registry."""
from __future__ import annotations

import copy
import json
import os
import stat
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.utils.config import Config, resolve_path
from src.utils.env import collect_env
from src.utils.hashing import sha256_file
from src.utils.reproducibility import sha256_tensor_dict


def sidecar_path(path: Path) -> Path:
    """JSON metadata file next to a checkpoint."""
    return path.with_suffix(".json")


def _make_read_only(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)


def save_checkpoint(model: nn.Module, path: str | Path, meta: dict[str, Any]) -> str:
    """Save the state dict and a sidecar JSON; make both read-only; return the state-dict SHA-256.

    ``meta`` should carry seed, config hash, split manifest hash and warm-up history.
    Refuses to overwrite an existing checkpoint.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; checkpoints are immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    state_hash = sha256_tensor_dict(state)
    torch.save(state, path)
    sidecar = {**meta, "state_dict_sha256": state_hash, "file_sha256": sha256_file(path),
               "env": collect_env()}
    sidecar_path(path).write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    _make_read_only(path)
    _make_read_only(sidecar_path(path))
    return state_hash


def load_checkpoint(path: str | Path, expected_hash: str, model: nn.Module | None = None,
                    device: str | torch.device = "cpu") -> nn.Module:
    """Load a checkpoint after verifying its state-dict SHA-256; raises ValueError on mismatch.

    If ``model`` is None an (untrained-weights) AlexNet with a 10-class head is built.
    """
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    actual = sha256_tensor_dict(state)
    if actual != expected_hash:
        raise ValueError(f"checkpoint hash mismatch for {path}: expected {expected_hash}, got {actual}")
    if model is None:
        from src.models.alexnet import build_model
        model = build_model(pretrained=False)
    model.load_state_dict(state)
    return model.to(device)


def clone_model(model: nn.Module) -> nn.Module:
    """Independent deep copy on the same device (parameters, buffers and requires_grad preserved)."""
    return copy.deepcopy(model)


# --- common-checkpoint registry (results/checkpoints_registry.json) ---

def baseline_path(cfg: Config, seed: int) -> Path:
    """checkpoints/baseline/seed{S}/common_start.pt"""
    return resolve_path(cfg.paths.baseline_dir) / f"seed{seed}" / "common_start.pt"


def _registry_path(cfg: Config) -> Path:
    return resolve_path(cfg.paths.registry)


def read_registry(cfg: Config) -> dict[str, Any]:
    path = _registry_path(cfg)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def register_checkpoint(cfg: Config, seed: int, path: Path, state_hash: str) -> None:
    """Record seed -> path and hashes; refuses to change an existing different entry."""
    registry = read_registry(cfg)
    root = resolve_path(".")
    stored = path.resolve().relative_to(root) if path.resolve().is_relative_to(root) else path.resolve()
    entry = {"path": str(stored), "state_dict_sha256": state_hash, "file_sha256": sha256_file(path)}
    old = registry.get(str(seed))
    if old and old["state_dict_sha256"] != state_hash:
        raise ValueError(f"seed {seed} already registered with a different hash")
    registry[str(seed)] = entry
    _registry_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    _registry_path(cfg).write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")


def require_common_checkpoint(cfg: Config, seed: int, device: str | torch.device = "cpu") -> tuple[nn.Module, str]:
    """Return (model, state-dict hash) for the registered common checkpoint of ``seed``.

    Exits with a clear error if the seed is unregistered, the file is missing, or any hash differs.
    """
    entry = read_registry(cfg).get(str(seed))
    if entry is None:
        raise SystemExit(f"ERROR: no common checkpoint registered for seed {seed}. "
                         f"Run: python main.py --experiment baseline --seed {seed}")
    path = resolve_path(entry["path"])
    if not path.is_file():
        raise SystemExit(f"ERROR: registered checkpoint missing: {path}")
    if sha256_file(path) != entry["file_sha256"]:
        raise SystemExit(f"ERROR: file hash mismatch for {path}; refusing to run")
    try:
        model = load_checkpoint(path, entry["state_dict_sha256"], device=device)
    except ValueError as err:
        raise SystemExit(f"ERROR: {err}") from err
    return model, entry["state_dict_sha256"]
