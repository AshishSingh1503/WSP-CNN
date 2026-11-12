"""Apply and revert validated edits on a CLONED model, with exact bookkeeping and safety checks."""
from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.utils.config import DslConfig
from src.utils.reproducibility import sha256_tensor_dict
from src.wsp.bitwise import differs
from src.wsp.edit_plan import EditPlan, RejectedEdit, TensorChange, layer_of, plan_edit
from src.wsp.edit_schema import Proposal

_PROTECTED: "weakref.WeakSet[nn.Module]" = weakref.WeakSet()


class EditError(Exception):
    """Base class for editor failures."""


class ProtectedModelError(EditError):
    """apply_edit was called on a protected (source / common-checkpoint) model."""


class NonFiniteError(EditError):
    """A parameter is NaN or inf."""


class CountMismatchError(EditError):
    """The measured number of changed scalars differs from the predicted number."""


def protect_model(model: nn.Module) -> nn.Module:
    """Mark ``model`` as read-only for the editor (the common checkpoint object); returns it."""
    _PROTECTED.add(model)
    return model


def is_protected(model: nn.Module) -> bool:
    return model in _PROTECTED


@dataclass
class EditRecord:
    """Everything needed to audit and exactly undo one applied edit."""

    proposal_id: str
    n_scalars: int
    per_tensor_counts: dict[str, int]
    per_layer_rel_norm: dict[str, float]
    global_rel_norm: float
    state_sha256: str
    npz_path: str | None
    changes: dict[str, TensorChange] = field(repr=False, default_factory=dict)


def assert_all_finite(model: nn.Module) -> None:
    """Raise NonFiniteError naming the first non-finite parameter."""
    for name, p in model.named_parameters():
        if not bool(torch.isfinite(p.detach()).all()):
            raise NonFiniteError(f"non-finite values in parameter {name}")


def _write(model: nn.Module, changes: dict[str, TensorChange], use_after: bool) -> None:
    params = dict(model.named_parameters())
    for name, c in changes.items():
        flat = params[name].detach().view(-1)
        flat[c.flat_idx.to(flat.device)] = (c.after if use_after else c.before).to(flat.device)


def _save_npz(path: Path, changes: dict[str, TensorChange]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for name, c in changes.items():
        arrays[f"{name}__idx"] = c.flat_idx.numpy()
        arrays[f"{name}__before"] = c.before.numpy()
        arrays[f"{name}__after"] = c.after.numpy()
    np.savez_compressed(path, **arrays)
    return str(path)


def _sq_norms(model: nn.Module) -> tuple[dict[str, float], float]:
    per_layer: dict[str, float] = {}
    for name, p in model.named_parameters():
        per_layer[layer_of(name)] = per_layer.get(layer_of(name), 0.0) + float(p.detach().double().pow(2).sum())
    return per_layer, sum(per_layer.values())


@torch.no_grad()
def apply_edit(model: nn.Module, proposal: Proposal, dsl: DslConfig, allowed_layers: tuple[str, ...],
               allowed_ops: tuple[str, ...], run_dir: Path | None = None) -> EditRecord:
    """Apply a (validated) proposal to ``model`` in place and return its EditRecord.

    Refuses protected models. After writing, checks the measured changed-scalar count equals the plan and
    that every parameter is finite; on any failure the edit is reverted before the error is raised.
    """
    if is_protected(model):
        raise ProtectedModelError("refusing to edit a protected source/common-checkpoint model; clone it first")
    try:
        plan: EditPlan = plan_edit(model, proposal, dsl, allowed_layers, allowed_ops)
    except RejectedEdit as rej:
        raise EditError(f"proposal not applicable: {rej}") from rej
    theta_layer, theta_global = _sq_norms(model)
    params = dict(model.named_parameters())
    originals = {n: params[n].detach().clone() for n in plan.changes}
    _write(model, plan.changes, use_after=True)
    try:
        measured = sum(int(differs(params[n].detach(), originals[n]).sum()) for n in plan.changes)
        if measured != plan.n_scalars:
            raise CountMismatchError(f"measured {measured} changed scalars, plan predicted {plan.n_scalars}")
        assert_all_finite(model)
    except EditError:
        _write(model, plan.changes, use_after=False)
        raise
    delta_layer: dict[str, float] = {}
    for n, c in plan.changes.items():
        delta_layer[layer_of(n)] = delta_layer.get(layer_of(n), 0.0) + float(((c.after.double() - c.before.double()) ** 2).sum())
    npz = _save_npz(run_dir / "edits" / f"{proposal.proposal_id}.npz", plan.changes) if run_dir else None
    return EditRecord(
        proposal_id=proposal.proposal_id, n_scalars=plan.n_scalars,
        per_tensor_counts={n: len(c.flat_idx) for n, c in plan.changes.items()},
        per_layer_rel_norm={k: (v ** 0.5) / max(theta_layer[k] ** 0.5, 1e-30) for k, v in delta_layer.items()},
        global_rel_norm=(sum(delta_layer.values()) ** 0.5) / max(theta_global ** 0.5, 1e-30),
        state_sha256=sha256_tensor_dict(model.state_dict()), npz_path=npz, changes=plan.changes)


@torch.no_grad()
def revert_edit(model: nn.Module, record: EditRecord) -> None:
    """Restore the exact before-values of an applied edit (bit-identical state dict)."""
    if is_protected(model):
        raise ProtectedModelError("refusing to modify a protected model")
    _write(model, record.changes, use_after=False)
