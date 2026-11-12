"""Turn a Proposal into an exact tensor-level EditPlan. Shared by the validator and the editor so that
every arm validates and applies edits through the same code."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from src.utils.config import DslConfig
from src.wsp.bitwise import differs
from src.wsp.edit_schema import (AddDelta, BiasShift, LowRankUpdate, NoEdit, Proposal,
                                 ScaleOutputChannel)


class RejectedEdit(Exception):
    """A proposal that cannot be planned; ``errors`` lists every problem found."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class TensorChange:
    """Sparse change to one parameter tensor: flat indices with exact before/after values."""

    name: str
    shape: tuple[int, ...]
    flat_idx: torch.Tensor
    before: torch.Tensor
    after: torch.Tensor


@dataclass
class EditPlan:
    """All scalar changes a proposal would make (relative to the model it was planned on)."""

    changes: dict[str, TensorChange] = field(default_factory=dict)

    @property
    def n_scalars(self) -> int:
        return sum(len(c.flat_idx) for c in self.changes.values())

    @property
    def delta_sq(self) -> float:
        return sum(float(((c.after.double() - c.before.double()) ** 2).sum()) for c in self.changes.values())


def layer_of(param_name: str) -> str:
    """'features.10.weight' -> 'features.10'."""
    return param_name.rsplit(".", 1)[0]


class _Planner:
    """Applies operations sequentially to working copies of the touched tensors."""

    def __init__(self, model: nn.Module, dsl: DslConfig, allowed_layers: tuple[str, ...],
                 allowed_ops: tuple[str, ...]) -> None:
        self.params = dict(model.named_parameters())
        self.modules = dict(model.named_modules())
        self.dsl, self.allowed_layers, self.allowed_ops = dsl, allowed_layers, allowed_ops
        self.work: dict[str, torch.Tensor] = {}
        self.errors: list[str] = []

    def tensor(self, name: str) -> torch.Tensor:
        if name not in self.work:
            self.work[name] = self.params[name].detach().clone()
        return self.work[name]

    def err(self, op: str, msg: str) -> None:
        self.errors.append(f"{op}: {msg}")

    def _layer_ok(self, op: str, layer: str, kinds: tuple[type, ...] = (nn.Conv2d, nn.Linear)) -> bool:
        module = self.modules.get(layer)
        if module is None or not isinstance(module, kinds):
            self.err(op, f"unknown layer {layer!r}")
            return False
        if layer not in self.allowed_layers:
            self.err(op, f"layer {layer!r} is not allowed in this experiment")
            return False
        return True

    def _unique_in_range(self, op: str, what: str, idx: list[int], size: int, limit: int) -> bool:
        ok = True
        if not idx:
            self.err(op, f"{what} is empty"); return False
        if len(idx) > limit:
            self.err(op, f"{len(idx)} {what} exceeds the limit {limit}"); ok = False
        if len(set(idx)) != len(idx):
            self.err(op, f"duplicate {what}"); ok = False
        if any(i < 0 or i >= size for i in idx):
            self.err(op, f"{what} out of range [0, {size})"); ok = False
        return ok

    def run(self, op: AddDelta | ScaleOutputChannel | BiasShift | LowRankUpdate | NoEdit) -> None:
        name = op.operation
        if name not in self.allowed_ops:
            self.err(name, "operation not permitted in this experiment")
        elif isinstance(op, AddDelta):
            self._add_delta(op)
        elif isinstance(op, ScaleOutputChannel):
            self._scale(op)
        elif isinstance(op, BiasShift):
            self._bias(op)
        elif isinstance(op, LowRankUpdate):
            self._low_rank(op)

    def _add_delta(self, op: AddDelta) -> None:
        cfg = self.dsl.add_delta
        if op.target_param not in self.params:
            return self.err("add_delta", f"unknown parameter {op.target_param!r}")
        if not self._layer_ok("add_delta", layer_of(op.target_param)):
            return
        shape = tuple(self.params[op.target_param].shape)
        if not op.indices or len(op.indices) > cfg.max_indices:
            return self.err("add_delta", f"need 1..{cfg.max_indices} indices, got {len(op.indices)}")
        if any(len(i) != len(shape) for i in op.indices):
            return self.err("add_delta", f"each index must have {len(shape)} coordinates for shape {list(shape)}")
        if any(c < 0 or c >= s for i in op.indices for c, s in zip(i, shape)):
            return self.err("add_delta", "index out of range")
        flat = np.ravel_multi_index(np.array(op.indices).T, shape)
        if len(np.unique(flat)) != len(flat):
            return self.err("add_delta", "duplicate indices")
        flat_t = torch.as_tensor(flat, dtype=torch.long)
        view = self.tensor(op.target_param).view(-1)
        cur = view[flat_t.to(view.device)].double().cpu()
        delta = torch.tensor(op.deltas, dtype=torch.float64)
        bound = torch.minimum(torch.full_like(cur, cfg.max_abs_delta),
                              cfg.rel_value_factor * cur.abs() + cfg.rel_value_offset)
        if bool((delta.abs() > bound).any()):
            return self.err("add_delta", f"|delta| exceeds min(max_abs_delta={cfg.max_abs_delta}, "
                                         f"{cfg.rel_value_factor}*|value|+{cfg.rel_value_offset})")
        view[flat_t.to(view.device)] = (cur + delta).to(view.dtype).to(view.device)

    def _scale(self, op: ScaleOutputChannel) -> None:
        cfg = self.dsl.scale_output_channel
        if not self._layer_ok("scale_output_channel", op.target_layer):
            return
        w = self.params[f"{op.target_layer}.weight"]
        if not self._unique_in_range("scale_output_channel", "channel_indices", op.channel_indices,
                                     w.shape[0], cfg.max_channels):
            return
        bad = [f for f in op.scale_factors
               if not (cfg.scale_min <= f <= cfg.scale_max or (cfg.allow_zero_channel and f == 0.0))]
        if bad:
            return self.err("scale_output_channel", f"scale factors {bad} outside [{cfg.scale_min}, "
                                                    f"{cfg.scale_max}]" + (" or 0.0" if cfg.allow_zero_channel else ""))
        ch = torch.tensor(op.channel_indices, dtype=torch.long)
        factors = torch.tensor(op.scale_factors, dtype=torch.float32)
        for suffix in ("weight", "bias"):
            t = self.tensor(f"{op.target_layer}.{suffix}")
            scale = factors.to(t.device).reshape(-1, *([1] * (t.dim() - 1)))
            t[ch.to(t.device)] = t[ch.to(t.device)] * scale

    def _bias(self, op: BiasShift) -> None:
        cfg = self.dsl.bias_shift
        if not self._layer_ok("bias_shift", op.target_layer):
            return
        bias = self.params.get(f"{op.target_layer}.bias")
        if bias is None:
            return self.err("bias_shift", f"layer {op.target_layer!r} has no bias")
        if not self._unique_in_range("bias_shift", "indices", op.indices, bias.shape[0], cfg.max_indices):
            return
        limit = (cfg.max_bias_shift_classifier if op.target_layer == cfg.classifier_layer
                 else cfg.max_bias_shift_other)
        if any(abs(s) > limit for s in op.shifts):
            return self.err("bias_shift", f"|shift| exceeds {limit} for layer {op.target_layer!r}")
        t = self.tensor(f"{op.target_layer}.bias")
        idx = torch.tensor(op.indices, dtype=torch.long, device=t.device)
        t[idx] = (t[idx].double() + torch.tensor(op.shifts, dtype=torch.float64, device=t.device)).to(t.dtype)

    def _low_rank(self, op: LowRankUpdate) -> None:
        cfg = self.dsl.low_rank_update
        layer = layer_of(op.target_param)
        if not op.target_param.endswith(".weight") or op.target_param not in self.params:
            return self.err("low_rank_update", f"{op.target_param!r} is not a known weight parameter")
        if not self._layer_ok("low_rank_update", layer, (nn.Linear,)):
            return
        w = self.params[op.target_param]
        rank = len(op.u)
        ok = rank <= cfg.max_rank
        if not ok:
            self.err("low_rank_update", f"rank {rank} exceeds max_rank {cfg.max_rank}")
        ok &= self._unique_in_range("low_rank_update", "rows", op.rows, w.shape[0], cfg.max_rows)
        ok &= self._unique_in_range("low_rank_update", "cols", op.cols, w.shape[1], cfg.max_cols)
        if not ok:
            return
        u = torch.tensor(op.u, dtype=torch.float64)
        v = torch.tensor(op.v, dtype=torch.float64)
        block = u.T @ v  # len(rows) x len(cols)
        norm_w = float(self.tensor(op.target_param).double().norm())
        if float(block.norm()) > cfg.max_rel_norm * norm_w:
            return self.err("low_rank_update", f"||update||_F={float(block.norm()):.4g} exceeds "
                                               f"{cfg.max_rel_norm}*||W||_F={cfg.max_rel_norm * norm_w:.4g}")
        t = self.tensor(op.target_param)
        rows = torch.tensor(op.rows, dtype=torch.long, device=t.device)
        cols = torch.tensor(op.cols, dtype=torch.long, device=t.device)
        sub = t[rows][:, cols].double() + block.to(t.device)
        t[rows.unsqueeze(1), cols.unsqueeze(0)] = sub.to(t.dtype)


def plan_edit(model: nn.Module, proposal: Proposal, dsl: DslConfig, allowed_layers: tuple[str, ...],
              allowed_ops: tuple[str, ...]) -> EditPlan:
    """Plan ``proposal`` on ``model`` without modifying it; raises RejectedEdit listing all problems."""
    planner = _Planner(model, dsl, allowed_layers, allowed_ops)
    for op in proposal.operations:
        planner.run(op)
    if planner.errors:
        raise RejectedEdit(planner.errors)
    plan = EditPlan()
    for name, new in planner.work.items():
        old = planner.params[name].detach()
        flat_new, flat_old = new.reshape(-1), old.reshape(-1)
        idx = torch.nonzero(differs(flat_new, flat_old)).reshape(-1)
        if len(idx):
            plan.changes[name] = TensorChange(name, tuple(old.shape), idx.cpu(), flat_old[idx].cpu().clone(),
                                              flat_new[idx].cpu().clone())
    only_no_edit = all(isinstance(op, NoEdit) for op in proposal.operations)
    if plan.n_scalars == 0 and not only_no_edit:
        raise RejectedEdit(["edit changes no parameter value (use no_edit instead)"])
    return plan
