"""Sparse checkpoints: a final state stored as the changed scalars relative to a start model.

An editing arm changes at most ~57k of 57M scalars, so saving the delta (and verifying the rebuilt state's
SHA-256) replaces a 230 MB checkpoint per run."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from src.utils.hashing import sha256_file
from src.utils.reproducibility import sha256_tensor_dict
from src.wsp.bitwise import differs

Overlay = dict[str, tuple[torch.Tensor, torch.Tensor]]  # param name -> (flat indices [long], values)


def overlay_between(start: nn.Module | Mapping[str, torch.Tensor], final: nn.Module | Mapping[str, torch.Tensor]) -> Overlay:
    """Scalars of ``final`` whose bits differ from ``start`` (CPU tensors)."""
    a = dict(start.named_parameters()) if isinstance(start, nn.Module) else dict(start)
    b = dict(final.named_parameters()) if isinstance(final, nn.Module) else dict(final)
    out: Overlay = {}
    for name, pa in a.items():
        fa, fb = pa.detach().cpu().reshape(-1), b[name].detach().cpu().reshape(-1)
        idx = torch.nonzero(differs(fa, fb)).reshape(-1)
        if len(idx):
            out[name] = (idx, fb[idx].clone())
    return out


def compose(first: Overlay, second: Overlay) -> Overlay:
    """Overlay equivalent to applying ``first`` then ``second`` (``second`` wins on shared positions)."""
    out: Overlay = {}
    for name in set(first) | set(second):
        parts = [o[name] for o in (first, second) if name in o]
        idx = torch.cat([p[0] for p in parts])
        val = torch.cat([p[1] for p in parts])
        order = torch.arange(len(idx))
        # keep the last occurrence of every index (later overlay wins)
        uniq, inverse = torch.unique(idx, return_inverse=True)
        last = torch.full((len(uniq),), -1, dtype=torch.long)
        last.scatter_reduce_(0, inverse, order, reduce="amax")
        out[name] = (uniq, val[last])
    return out


def realize(base_state: Mapping[str, torch.Tensor], overlay: Overlay) -> dict[str, torch.Tensor]:
    """Full state dict = ``base_state`` with ``overlay`` applied (copies only the touched tensors)."""
    state = dict(base_state)
    for name, (idx, val) in overlay.items():
        t = state[name].detach().clone().reshape(-1)
        t[idx] = val
        state[name] = t.reshape(base_state[name].shape)
    return state


def state_hash_with(base_state: Mapping[str, torch.Tensor], overlay: Overlay) -> str:
    """SHA-256 of the state dict obtained by applying ``overlay`` to ``base_state``."""
    return sha256_tensor_dict(realize(base_state, overlay))


def _read_only(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)


def save_delta(path: Path, start: nn.Module, final: nn.Module, meta: dict[str, Any]) -> dict[str, Any]:
    """Write ``path`` (.npz of changed scalars) and a sidecar JSON; both read-only. Returns hashes."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; checkpoints are immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_between(start, final)
    arrays: dict[str, np.ndarray] = {}
    for name, (idx, val) in overlay.items():
        arrays[f"{name}__idx"] = idx.numpy()
        arrays[f"{name}__val"] = val.numpy()
    np.savez_compressed(path, **arrays)
    info = {"final_state_sha256": sha256_tensor_dict(final.state_dict()),
            "start_state_sha256": sha256_tensor_dict(start.state_dict()),
            "n_changed": sum(len(i) for i, _ in overlay.values()), "file_sha256": sha256_file(path)}
    path.with_suffix(".json").write_text(json.dumps({**meta, **info}, indent=2, default=str), encoding="utf-8")
    _read_only(path)
    _read_only(path.with_suffix(".json"))
    return info


def load_delta(path: Path) -> Overlay:
    """Read an overlay written by ``save_delta``."""
    out: Overlay = {}
    with np.load(Path(path)) as data:
        for key in data.files:
            if key.endswith("__idx"):
                name = key[:-5]
                out[name] = (torch.from_numpy(data[key].astype(np.int64)), torch.from_numpy(data[f"{name}__val"]))
    return out


class SwitchingModel:
    """A model on ``device`` that can switch between variants defined as overlays on one base state.

    Switching restores the previous variant's positions to the base values and writes the new ones, so
    evaluating many variants costs a few sparse writes instead of a model copy each.
    """

    def __init__(self, model: nn.Module, base_state: Mapping[str, torch.Tensor], device: torch.device | str) -> None:
        self.model, self.device = model.to(device).eval(), device
        self.base = {n: t.detach().cpu().clone().reshape(-1) for n, t in base_state.items()}
        self._active: Overlay = {}
        self._params = dict(self.model.named_parameters())

    @torch.no_grad()
    def set_variant(self, overlay: Overlay | None) -> None:
        for name, (idx, _) in self._active.items():
            flat = self._params[name].detach().view(-1)
            flat[idx.to(self.device)] = self.base[name][idx.cpu()].to(self.device)
        self._active = overlay or {}
        for name, (idx, val) in self._active.items():
            flat = self._params[name].detach().view(-1)
            flat[idx.to(self.device)] = val.to(self.device)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)
