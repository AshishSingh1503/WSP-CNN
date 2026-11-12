"""E7: gradient-heuristic search without an LLM (Hybrid-style gradient information, hand-written rule)."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from src.diagnostics.gradient_analyzer import raw_gradients
from src.utils.config import Config
from src.wsp.llm_agent import AgentResult
from src.wsp.random_proposals import wrap

HEAD = "classifier.6"


class GradHeuristicAgent:
    """Proposes, per iteration, 3 channel-scaling candidates (alpha grid) and 1 head bias-shift candidate.

    - Rank every output channel/row of every allowed layer by ||grad W_c||_2 / sqrt(numel_c) on the diagnostic subset.
    - Scale candidate: for the top-k channels, factor = clip(1 - alpha * sign(<grad_c, w_c>)) (a first-order
      descent step on the channel's scale), grouped by layer, within the DSL bounds.
    - Bias candidate: classifier.6 bias shifts = -alpha * grad_b / max|grad_b|, within the DSL bounds.
    - If the model did not change since the last iteration (nothing accepted), move down the ranking so
      the same candidates are not proposed again.
    No LLM client exists on this object.
    """

    needs_packet = False

    def __init__(self, cfg: Config, allowed_layers: tuple[str, ...], allowed_ops: tuple[str, ...],
                 diag_batches: list[Any], device: torch.device | str) -> None:
        self.cfg, self.h, self.dsl = cfg, cfg.grad_heuristic, cfg.dsl
        self.layers, self.ops, self.diag, self.device = tuple(allowed_layers), tuple(allowed_ops), diag_batches, device
        self.all_usages: list[Any] = []
        self._last_model: nn.Module | None = None
        self._offset = 0

    def _ranked_channels(self, model: nn.Module, grads: dict[str, torch.Tensor]) -> list[tuple[float, str, int, float, int]]:
        rows = []
        for name, module in model.named_modules():
            if name in self.layers and isinstance(module, (nn.Conv2d, nn.Linear)) and f"{name}.weight" in grads:
                g, w = grads[f"{name}.weight"].double(), module.weight.detach().cpu().double()
                per = g[0].numel()
                norm = g.reshape(g.shape[0], -1).norm(dim=1) / math.sqrt(per)
                dot = (g * w).reshape(g.shape[0], -1).sum(dim=1)
                for c in range(g.shape[0]):
                    rows.append((float(norm[c]), name, c, float(dot[c]), per + 1))
        rows.sort(key=lambda r: (-r[0], r[1], r[2]))
        return rows

    def _scale_candidate(self, ranked: list[tuple[float, str, int, float, int]], alpha: float, remaining: int, pid: str) -> dict[str, Any] | None:
        c, k = self.dsl.scale_output_channel, self.h.top_k_channels
        chosen: dict[str, list[tuple[int, float]]] = {}
        used = 0
        for _, layer, ch, dot, size in ranked[self._offset * k:]:
            if dot == 0.0 or used + size > remaining:
                continue
            if layer not in chosen and len(chosen) >= self.dsl.max_ops_per_proposal:
                continue
            factor = min(c.scale_max, max(c.scale_min, 1.0 - alpha * math.copysign(1.0, dot)))
            chosen.setdefault(layer, []).append((ch, factor))
            used += size
            if sum(len(v) for v in chosen.values()) >= k:
                break
        if not chosen:
            return None
        ops = [{"operation": "scale_output_channel", "target_layer": layer, "channel_indices": [i for i, _ in sorted(v)],
                "scale_factors": [f for _, f in sorted(v)]} for layer, v in chosen.items()]
        return wrap(pid, ops, reason=f"grad heuristic scale alpha={alpha}")

    def _bias_candidate(self, grads: dict[str, torch.Tensor], alpha: float, pid: str) -> dict[str, Any] | None:
        if HEAD not in self.layers or "bias_shift" not in self.ops or f"{HEAD}.bias" not in grads:
            return None
        g = grads[f"{HEAD}.bias"].double()
        peak = float(g.abs().max())
        if peak == 0.0:
            return None
        limit = self.dsl.bias_shift.max_bias_shift_classifier
        shifts = (-alpha * g / peak).clamp(-limit, limit)
        keep = [i for i in range(len(shifts)) if abs(float(shifts[i])) > 1e-9]
        return wrap(pid, [{"operation": "bias_shift", "target_layer": HEAD, "indices": keep,
                           "shifts": [float(shifts[i]) for i in keep]}], reason=f"grad heuristic bias alpha={alpha}")

    def propose(self, packet_json: str, model: nn.Module, iteration: int, ctx: dict[str, Any] | None = None) -> AgentResult:
        remaining = (ctx or {}).get("remaining_edit_budget", self.cfg.model.expected_total_params)
        self._offset = self._offset + 1 if model is self._last_model else 0
        self._last_model = model
        grads, _, _ = raw_gradients(model, self.diag, self.device)
        ranked = self._ranked_channels(model, grads) if "scale_output_channel" in self.ops else []
        proposals: list[dict[str, Any] | None] = []
        if ranked:
            proposals += [self._scale_candidate(ranked, a, remaining, f"e7_i{iteration}_s{j}") for j, a in enumerate(self.h.alphas)]
        if "bias_shift" in self.ops:
            alpha = self.h.bias_alphas[(iteration - 1) % len(self.h.bias_alphas)]
            proposals.append(self._bias_candidate(grads, alpha, f"e7_i{iteration}_b"))
        k = self.cfg.budgets.candidates_per_iteration
        return AgentResult(proposals=[p for p in proposals if p is not None][:k])
