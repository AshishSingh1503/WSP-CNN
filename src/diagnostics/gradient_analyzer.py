"""Gradient diagnostics on the diagnostic subset. HYBRID WSP ONLY: the Pure arm must never import this."""
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from src.diagnostics.gradient_schema import (ChannelGradient, ClassPairGradient, ClassRowGradient,
                                             GradientDiagnostics, LayerGradient)
from src.utils.config import DiagnosticsConfig

HEAD_LAYER = "classifier.6"


def _layer_gradients(model: nn.Module, topk: int) -> list[LayerGradient]:
    out = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight.grad is not None:
            g = module.weight.grad.detach().cpu().double()
            norms = g.reshape(g.shape[0], -1).norm(dim=1)
            order = torch.argsort(norms, descending=True, stable=True)[:topk]
            out.append(LayerGradient(layer=name, normalised_grad_norm=float(g.norm() / g.numel() ** 0.5),
                                     top_channels=[ChannelGradient(index=int(i), grad_norm=float(norms[i]))
                                                   for i in order]))
    return out


def _head_summary(grad: torch.Tensor, n_pairs: int) -> tuple[list[ClassRowGradient], list[ClassPairGradient]]:
    g = grad.detach().cpu().double()
    rows = [ClassRowGradient(class_index=i, mean_sign=float(g[i].sign().mean()), grad_norm=float(g[i].norm()))
            for i in range(g.shape[0])]
    unit = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos = unit @ unit.T
    pairs = [(float(cos[i, j]), i, j) for i in range(len(g)) for j in range(i + 1, len(g))]
    pairs.sort(key=lambda t: (-abs(t[0]), t[1], t[2]))
    return rows, [ClassPairGradient(class_a=i, class_b=j, cosine=c) for c, i, j in pairs[:n_pairs]]


def analyze_gradients(model: nn.Module, batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
                      device: str | torch.device, cfg: DiagnosticsConfig) -> GradientDiagnostics:
    """Mean-CE gradients over all batches (eval mode); leaves every ``param.grad`` as None afterwards."""
    batches = list(batches)
    n_total = sum(len(y) for _, y in batches)
    was_training = model.training
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
        p.grad = None
    loss_total = 0.0
    try:
        for images, labels in batches:
            logits = model(images.to(device))
            loss = nn.functional.cross_entropy(logits.float(), labels.to(device), reduction="sum") / n_total
            loss.backward()
            loss_total += loss.item()
        layers = _layer_gradients(model, cfg.grad_topk)
        rows, pairs = _head_summary(dict(model.named_modules())[HEAD_LAYER].weight.grad, cfg.grad_top_pairs)
    finally:
        for n, p in model.named_parameters():
            p.grad = None
            p.requires_grad_(flags[n])
        model.train(was_training)
    return GradientDiagnostics(n_images=n_total, loss=loss_total, layers=layers,
                               classifier6_rows=rows, classifier6_pairs=pairs)
