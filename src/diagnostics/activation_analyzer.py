"""Channel-wise activation statistics over the whole diagnostic subset (forward hooks, no autograd)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from src.utils.config import DiagnosticsConfig

HOOKED = (nn.Conv2d, nn.ReLU, nn.Linear)


@dataclass
class _Accumulator:
    """Running per-channel totals (float64) for one hooked module."""

    kind: str
    count: int = 0
    sum: torch.Tensor | None = None
    sumsq: torch.Tensor | None = None
    sumabs: torch.Tensor | None = None
    inactive: torch.Tensor | None = None

    def update(self, x: torch.Tensor) -> None:
        dims = (0, 2, 3) if x.dim() == 4 else (0,)
        positions = x.numel() // x.shape[1]
        x = x.double()
        parts = (x.sum(dims), (x * x).sum(dims), x.abs().sum(dims), (x <= 0).double().sum(dims))
        if self.sum is None:
            self.sum, self.sumsq, self.sumabs, self.inactive = parts
        else:
            self.sum += parts[0]; self.sumsq += parts[1]; self.sumabs += parts[2]; self.inactive += parts[3]
        self.count += positions


class ActivationHooks:
    """Context manager registering forward hooks on Conv2d, ReLU and Linear modules; always removes them."""

    def __init__(self, model: nn.Module, pooled_layers: Iterable[str] = ()) -> None:
        self.model, self.pooled_layers = model, set(pooled_layers)
        self.acc: dict[str, _Accumulator] = {}
        self.pooled: dict[str, list[torch.Tensor]] = {n: [] for n in self.pooled_layers}
        self._handles: list[Any] = []

    def __enter__(self) -> "ActivationHooks":
        for name, module in self.model.named_modules():
            if isinstance(module, HOOKED):
                self._handles.append(module.register_forward_hook(self._make_hook(name, module)))
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, name: str, module: nn.Module):
        acc = self.acc.setdefault(name, _Accumulator(kind=type(module).__name__))

        def hook(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> None:
            x = out.detach()  # reductions happen now: a following in-place ReLU cannot affect them
            acc.update(x)
            if name in self.pooled_layers:
                self.pooled[name].append((x.mean(dim=(2, 3)) if x.dim() == 4 else x).double().cpu())
        return hook


@dataclass
class ActivationReport:
    """Per-layer channel statistics (numpy) and class-selective channels."""

    layers: dict[str, dict[str, Any]] = field(default_factory=dict)
    selectivity: dict[str, dict[int, list[dict[str, float]]]] = field(default_factory=dict)


def class_selectivity(pooled: torch.Tensor, labels: torch.Tensor, num_classes: int,
                      top: int) -> dict[int, list[dict[str, float]]]:
    """Top channels per class by (class mean - other mean) / pooled std over per-image channel values."""
    out: dict[int, list[dict[str, float]]] = {}
    for k in range(num_classes):
        in_k = labels == k
        a, b = pooled[in_k], pooled[~in_k]
        dof = max(1, len(a) + len(b) - 2)
        var = ((len(a) - 1) * a.var(dim=0) + (len(b) - 1) * b.var(dim=0)) / dof
        sel = (a.mean(0) - b.mean(0)) / (var.sqrt() + 1e-8)
        order = torch.argsort(sel, descending=True, stable=True)[:top]
        out[k] = [{"channel": int(c), "selectivity": float(sel[c]), "class_mean": float(a.mean(0)[c])}
                  for c in order]
    return out


@torch.no_grad()
def analyze_activations(model: nn.Module, batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
                        device: str | torch.device, cfg: DiagnosticsConfig,
                        num_classes: int = 10) -> ActivationReport:
    """Accumulate statistics over ALL batches (eval mode, dropout off) and finalise them."""
    model.eval()
    labels_all: list[torch.Tensor] = []
    with ActivationHooks(model, cfg.selectivity_layers) as hooks:
        for images, labels in batches:
            model(images.to(device))
            labels_all.append(labels.cpu())
    report = ActivationReport()
    for name, acc in hooks.acc.items():
        n = acc.count
        mean = acc.sum / n
        var = (acc.sumsq / n - mean * mean).clamp_min(0)
        report.layers[name] = {"kind": acc.kind, "positions": n, "mean": mean.cpu().numpy(),
                               "std": var.sqrt().cpu().numpy(), "mean_abs": (acc.sumabs / n).cpu().numpy(),
                               "inactive_fraction": (acc.inactive / n).cpu().numpy()}
    labels = torch.cat(labels_all)
    for name in cfg.selectivity_layers:
        report.selectivity[name] = class_selectivity(torch.cat(hooks.pooled[name]), labels, num_classes,
                                                     cfg.selectivity_top)
    return report


def summarize_activations(report: ActivationReport, cfg: DiagnosticsConfig) -> dict[str, Any]:
    """Compact JSON-able summary: averages, persistently dead vs merely sparse channels, selectivity."""
    layers: dict[str, Any] = {}
    for name, d in report.layers.items():
        inactive = d["inactive_fraction"]
        dead = np.flatnonzero(inactive >= cfg.dead_threshold)
        sparse = np.flatnonzero((inactive >= cfg.sparse_threshold) & (inactive < cfg.dead_threshold))
        entry: dict[str, Any] = {"type": d["kind"], "channels": int(len(inactive)),
                                 "mean": float(d["mean"].mean()), "mean_abs": float(d["mean_abs"].mean()),
                                 "inactive_fraction": float(inactive.mean())}
        if d["kind"] != "Conv2d":  # pre-ReLU conv outputs duplicate the ReLU statistics
            entry.update(n_dead=int(len(dead)), dead_channels=dead[:cfg.max_listed_channels].tolist(),
                         n_sparse=int(len(sparse)), sparse_channels=sparse[:cfg.max_listed_channels].tolist())
        layers[name] = entry
    selectivity = {layer: {str(k): v for k, v in per_class.items()}
                   for layer, per_class in report.selectivity.items()}
    return {"layers": layers, "class_selective_channels": selectivity}
