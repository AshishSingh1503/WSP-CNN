"""Parameter-change statistics between two models (edit size and relative update norm)."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _layer_of(name: str) -> str:
    """'features.10.weight' -> 'features.10'."""
    return name.rsplit(".", 1)[0]


def _entry(n_changed: int, numel: int, delta_sq: float, base_sq: float) -> dict[str, Any]:
    rel = (delta_sq ** 0.5) / (base_sq ** 0.5) if base_sq > 0 else 0.0
    return {"n_changed": n_changed, "numel": numel, "rel_norm": rel}


@torch.no_grad()
def diff_stats(model_a: nn.Module, model_b: nn.Module) -> dict[str, Any]:
    """Compare ``model_b`` against reference ``model_a`` (PROTOCOL.md Section 7).

    Edit size = scalars whose value differs bitwise; relative norm = ||b - a||_2 / ||a||_2,
    reported globally, per layer (weight + bias) and per tensor.
    """
    params_b = dict(model_b.named_parameters())
    per_tensor: dict[str, dict[str, Any]] = {}
    sums: dict[str, list[float]] = {}  # layer -> [n_changed, numel, delta_sq, base_sq]
    total = [0, 0, 0.0, 0.0]
    for name, a in model_a.named_parameters():
        a_cpu, b_cpu = a.detach().cpu(), params_b[name].detach().cpu()
        delta = (b_cpu.double() - a_cpu.double())
        vals = [int((a_cpu != b_cpu).sum()), a_cpu.numel(), float((delta ** 2).sum()),
                float((a_cpu.double() ** 2).sum())]
        per_tensor[name] = _entry(*vals)
        layer = sums.setdefault(_layer_of(name), [0, 0, 0.0, 0.0])
        for acc in (layer, total):
            for i, v in enumerate(vals):
                acc[i] += v
    return {"n_changed": int(total[0]), "n_total": int(total[1]),
            "global_rel_norm": _entry(*total)["rel_norm"],
            "per_layer": {k: _entry(*v) for k, v in sums.items()},
            "per_tensor": per_tensor}
