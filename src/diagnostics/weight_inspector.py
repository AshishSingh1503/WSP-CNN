"""Weight statistics per tensor and per output unit (read-only; no autograd)."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


CHUNK = 1 << 22  # elements per float64 chunk (keeps the 37M-parameter layer from being copied whole)


def tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    """Shape, numel, mean, std (population), min, max, L1, L2, zero fraction, fraction with |w| > 3 std."""
    flat = t.detach().cpu().reshape(-1)
    n = flat.numel()
    total = sq = l1 = 0.0
    zeros = 0
    for chunk in flat.split(CHUNK):
        x = chunk.double()
        total += float(x.sum()); sq += float((x * x).sum()); l1 += float(x.abs().sum())
        zeros += int((x == 0).sum())
    mean = total / n
    std = max(sq / n - mean * mean, 0.0) ** 0.5
    beyond = sum(int((chunk.double().abs() > 3 * std).sum()) for chunk in flat.split(CHUNK))
    return {"shape": list(t.shape), "numel": n, "mean": mean, "std": std, "min": float(flat.min()),
            "max": float(flat.max()), "l1": l1, "l2": sq ** 0.5, "zero_fraction": zeros / n,
            "frac_abs_gt_3std": beyond / n}


def unit_stats(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """L2 norm, mean and std per output unit (conv output channel / linear row), computed row-block-wise."""
    flat = weight.detach().cpu().reshape(weight.shape[0], -1)
    block = max(1, CHUNK // flat.shape[1])
    l2, mean, std = [], [], []
    for rows in flat.split(block):
        x = rows.double()
        l2.append(x.norm(dim=1)); mean.append(x.mean(dim=1)); std.append(x.std(dim=1, unbiased=False))
    return {"l2": torch.cat(l2), "mean": torch.cat(mean), "std": torch.cat(std)}


def _units(stats: dict[str, torch.Tensor], order: torch.Tensor) -> list[dict[str, float]]:
    return [{"index": int(i), "l2": float(stats["l2"][i]), "mean": float(stats["mean"][i]),
             "std": float(stats["std"][i])} for i in order]


def inspect_weights(model: nn.Module, topk: int) -> dict[str, Any]:
    """Per-tensor stats for every parameter, plus top-k / bottom-k output units by L2 norm per layer."""
    with torch.no_grad():
        tensors = {name: tensor_stats(p) for name, p in model.named_parameters()}
        units: dict[str, Any] = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                st = unit_stats(module.weight)
                k = min(topk, len(st["l2"]))
                desc = torch.argsort(st["l2"], descending=True, stable=True)
                asc = torch.argsort(st["l2"], descending=False, stable=True)
                units[name] = {"n_units": len(st["l2"]), "top": _units(st, desc[:k]),
                               "bottom": _units(st, asc[:k])}
    return {"tensors": tensors, "units": units}
