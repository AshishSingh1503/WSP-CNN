"""Random valid proposals drawn from the edit DSL (E5, E6-RS and the building blocks of E4).

No LLM, no gradients, no validation feedback: only the model's shapes and current values (needed for the
value-relative add_delta bound) and the DSL bounds from config are used."""
from __future__ import annotations

import zlib
from typing import Any

import numpy as np
import torch
from torch import nn

from src.utils.config import Config
from src.wsp.edit_schema import AddDelta, BiasShift, LowRankUpdate, ScaleOutputChannel
from src.wsp.llm_agent import AgentResult

OP_NAMES = ("add_delta", "scale_output_channel", "bias_shift", "low_rank_update")


def arm_rng(seed: int, arm: str) -> np.random.Generator:
    """Deterministic generator for (seed, arm)."""
    return np.random.default_rng([seed, zlib.crc32(arm.encode("utf-8"))])


def wrap(proposal_id: str, ops: list[dict[str, Any]], reason: str = "random") -> dict[str, Any]:
    """Proposal envelope with neutral metadata."""
    return {"proposal_id": proposal_id, "reason": reason, "evidence": [], "expected_effect": "n/a",
            "confidence": 0.0, "risk": "n/a", "operations": ops}


class RandomProposer:
    """Samples ops with sizes/magnitudes from ``cfg.random_search`` inside the ``cfg.dsl`` bounds."""

    def __init__(self, cfg: Config, allowed_layers: tuple[str, ...], allowed_ops: tuple[str, ...],
                 rng: np.random.Generator) -> None:
        self.cfg, self.rs, self.dsl, self.rng = cfg, cfg.random_search, cfg.dsl, rng
        self.layers, self.ops = tuple(allowed_layers), tuple(o for o in allowed_ops if o in OP_NAMES)

    # -- helpers --
    def _modules(self, model: nn.Module, kinds: tuple[type, ...] = (nn.Conv2d, nn.Linear)) -> dict[str, nn.Module]:
        return {n: m for n, m in model.named_modules() if n in self.layers and isinstance(m, kinds)}

    def _uniform(self, lo_hi: tuple[float, float]) -> float:
        return float(self.rng.uniform(*lo_hi))

    def bias_limit(self, layer: str) -> float:
        b = self.dsl.bias_shift
        return b.max_bias_shift_classifier if layer == b.classifier_layer else b.max_bias_shift_other

    # -- op builders (explicit sizes; magnitudes are fractions of the bounds) --
    def make_add_delta(self, model: nn.Module, layer: str, n: int, mag: tuple[float, float], use_weight: bool) -> dict[str, Any]:
        name = f"{layer}.{'weight' if use_weight else 'bias'}"
        t = dict(model.named_parameters())[name].detach()
        n = min(n, t.numel(), self.dsl.add_delta.max_indices)
        flat = self.rng.choice(t.numel(), n, replace=False)
        cur = t.reshape(-1)[torch.as_tensor(flat, device=t.device)].double().abs().cpu().numpy()
        c = self.dsl.add_delta
        bound = np.minimum(c.max_abs_delta, c.rel_value_factor * cur + c.rel_value_offset) * self.rs.safety
        delta = self.rng.choice([-1.0, 1.0], n) * bound * self.rng.uniform(*mag, size=n)
        idx = np.stack(np.unravel_index(flat, tuple(t.shape)), axis=1).tolist()
        return AddDelta(operation="add_delta", target_param=name, indices=idx, deltas=delta.tolist()).model_dump()

    def make_scale(self, model: nn.Module, layer: str, k: int, dev_scale: float = 1.0) -> dict[str, Any]:
        c = self.dsl.scale_output_channel
        out = self._modules(model)[layer].weight.shape[0]
        k = max(1, min(k, out, c.max_channels))
        ch = self.rng.choice(out, k, replace=False)
        factors = []
        for _ in range(k):
            down = bool(self.rng.integers(2))
            room = (1.0 - c.scale_min) if down else (c.scale_max - 1.0)
            dev = self.rng.uniform(min(self.rs.scale_min_deviation, room * self.rs.safety), room * self.rs.safety) * dev_scale
            factors.append(1.0 - dev if down else 1.0 + dev)
        return ScaleOutputChannel(operation="scale_output_channel", target_layer=layer,
                                  channel_indices=sorted(int(i) for i in ch), scale_factors=factors).model_dump()

    def make_bias_shift(self, model: nn.Module, layer: str, n: int, mag: tuple[float, float]) -> dict[str, Any]:
        out = self._modules(model)[layer].weight.shape[0]
        n = max(1, min(n, out, self.dsl.bias_shift.max_indices))
        idx = np.sort(self.rng.choice(out, n, replace=False))
        shifts = self.rng.choice([-1.0, 1.0], n) * self.bias_limit(layer) * self.rs.safety * self.rng.uniform(*mag, size=n)
        return BiasShift(operation="bias_shift", target_layer=layer, indices=[int(i) for i in idx],
                         shifts=shifts.tolist()).model_dump()

    def make_low_rank(self, model: nn.Module, layer: str, rows: int, cols: int, rank: int, norm_fraction: float) -> dict[str, Any]:
        c = self.dsl.low_rank_update
        w = self._modules(model, (nn.Linear,))[layer].weight
        rows, cols = max(1, min(rows, w.shape[0], c.max_rows)), max(1, min(cols, w.shape[1], c.max_cols))
        rank = max(1, min(rank, c.max_rank))
        r_idx = sorted(int(i) for i in self.rng.choice(w.shape[0], rows, replace=False))
        c_idx = sorted(int(i) for i in self.rng.choice(w.shape[1], cols, replace=False))
        u, v = self.rng.normal(size=(rank, rows)), self.rng.normal(size=(rank, cols))
        target = norm_fraction * c.max_rel_norm * self.rs.safety * float(w.detach().double().norm())
        u *= target / max(float(np.linalg.norm(u.T @ v)), 1e-30)
        return LowRankUpdate(operation="low_rank_update", target_param=f"{layer}.weight", rows=r_idx, cols=c_idx,
                             u=u.tolist(), v=v.tolist()).model_dump()

    # -- random sampling --
    def _size_estimate(self, model: nn.Module, op: dict[str, Any]) -> int:
        mods = self._modules(model)
        if op["operation"] == "scale_output_channel":
            w = mods[op["target_layer"]].weight
            return len(op["channel_indices"]) * (w.numel() // w.shape[0] + 1)
        if op["operation"] == "low_rank_update":
            return len(op["rows"]) * len(op["cols"])
        return len(op.get("indices", []))

    def sample_op(self, model: nn.Module) -> dict[str, Any]:
        """One random op of random type on a random allowed layer."""
        rs = self.rs
        feasible = [o for o in self.ops if o != "low_rank_update" or self._modules(model, (nn.Linear,))]
        op = str(self.rng.choice(feasible))
        if op == "low_rank_update":
            layer = str(self.rng.choice(sorted(self._modules(model, (nn.Linear,)))))
            return self.make_low_rank(model, layer, int(self.rng.integers(1, rs.low_rank_max_rows + 1)),
                                      int(self.rng.integers(1, rs.low_rank_max_cols + 1)),
                                      int(self.rng.integers(1, rs.low_rank_max_rank + 1)), self._uniform(rs.low_rank_norm_fraction))
        layer = str(self.rng.choice(sorted(self._modules(model))))
        if op == "add_delta":
            return self.make_add_delta(model, layer, int(self.rng.integers(1, rs.add_delta_max_entries + 1)),
                                       rs.add_delta_magnitude, bool(self.rng.random() < rs.param_weight_prob))
        if op == "scale_output_channel":
            return self.make_scale(model, layer, int(self.rng.integers(1, rs.scale_max_channels + 1)))
        return self.make_bias_shift(model, layer, int(self.rng.integers(1, rs.bias_max_indices + 1)), rs.bias_magnitude)

    def sample(self, model: nn.Module, remaining_scalars: int, proposal_id: str) -> dict[str, Any]:
        """A random proposal of 1..max_ops ops whose estimated size fits the remaining edit budget."""
        n_ops = int(self.rng.integers(1, self.rs.max_ops + 1))
        ops: list[dict[str, Any]] = []
        used = 0
        for _ in range(n_ops):
            for _try in range(self.rs.max_sampling_tries):
                op = self.sample_op(model)
                size = self._size_estimate(model, op)
                if used + size <= remaining_scalars:
                    ops.append(op)
                    used += size
                    break
        if not ops:
            return wrap(proposal_id, [{"operation": "no_edit", "reason": "no random edit fits the remaining budget"}])
        return wrap(proposal_id, ops)


class RandomSearchAgent:
    """Drop-in proposer for the controller: K random proposals per iteration, no packet, no LLM."""

    needs_packet = False

    def __init__(self, cfg: Config, allowed_layers: tuple[str, ...], allowed_ops: tuple[str, ...], seed: int, arm: str) -> None:
        self.cfg, self.proposer = cfg, RandomProposer(cfg, allowed_layers, allowed_ops, arm_rng(seed, arm))
        self.all_usages: list[Any] = []

    def propose(self, packet_json: str, model: nn.Module, iteration: int, ctx: dict[str, Any] | None = None) -> AgentResult:
        remaining = (ctx or {}).get("remaining_edit_budget", self.cfg.model.expected_total_params)
        k = self.cfg.budgets.candidates_per_iteration
        return AgentResult(proposals=[self.proposer.sample(model, remaining, f"rs_i{iteration}_{j}") for j in range(k)])
