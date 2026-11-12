"""Validate LLM proposals against the DSL, the experiment's permissions and the remaining budget.

``validate_proposal`` never raises on bad proposals: every problem is returned in ``ValidationResult.errors``."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from src.utils.config import DslConfig
from src.wsp.budget import BudgetTracker
from src.wsp.edit_plan import EditPlan, RejectedEdit, plan_edit
from src.wsp.edit_schema import NoEdit, Proposal, parse_proposal

ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass
class ValidationContext:
    """What this experiment allows and how much budget is left."""

    dsl: DslConfig
    allowed_layers: tuple[str, ...]
    allowed_ops: tuple[str, ...]
    edit_budget: int
    tracker: BudgetTracker | None = None


@dataclass
class ValidationResult:
    """Outcome of validation. ``n_scalars`` is the size of this edit relative to the current model;
    ``cumulative_after`` the changed-scalar count vs the common checkpoint after applying it."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    n_scalars: int = 0
    rel_norm: float = 0.0
    cumulative_after: int = 0
    proposal: Proposal | None = field(default=None, repr=False)
    plan: EditPlan | None = field(default=None, repr=False)


def _global_norm(model: nn.Module) -> float:
    return sum(float(p.detach().double().pow(2).sum()) for p in model.parameters()) ** 0.5


def _text_errors(p: Proposal, ctx: ValidationContext) -> list[str]:
    lim, errors = ctx.dsl.limits, []
    if not p.proposal_id or len(p.proposal_id) > lim.max_id_chars or not ID_PATTERN.match(p.proposal_id):
        errors.append(f"proposal_id must match [A-Za-z0-9_.-] and be 1..{lim.max_id_chars} chars")
    if len(p.reason) > lim.max_reason_chars:
        errors.append(f"reason longer than {lim.max_reason_chars} chars")
    for name in ("expected_effect", "risk"):
        if len(getattr(p, name)) > lim.max_text_chars:
            errors.append(f"{name} longer than {lim.max_text_chars} chars")
    if len(p.evidence) > lim.max_evidence_items or any(len(e) > lim.max_evidence_item_chars for e in p.evidence):
        errors.append(f"evidence limited to {lim.max_evidence_items} items of <= {lim.max_evidence_item_chars} chars")
    n_ops = len(p.operations)
    if n_ops < 1 or n_ops > ctx.dsl.max_ops_per_proposal:
        errors.append(f"a proposal needs 1..{ctx.dsl.max_ops_per_proposal} operations, got {n_ops}")
    if n_ops > 1 and any(isinstance(op, NoEdit) for op in p.operations):
        errors.append("no_edit must be the only operation in a proposal")
    return errors


def validate_proposal(raw: str | bytes | dict[str, Any] | Proposal, model: nn.Module,
                      ctx: ValidationContext) -> ValidationResult:
    """Parse and validate ``raw`` against ``model`` (read-only). Never raises on bad input."""
    proposal, errors = (raw, []) if isinstance(raw, Proposal) else parse_proposal(raw)
    if proposal is None:
        return ValidationResult(False, errors)
    errors = _text_errors(proposal, ctx)
    plan: EditPlan | None = None
    try:
        with torch.no_grad():
            plan = plan_edit(model, proposal, ctx.dsl, ctx.allowed_layers, ctx.allowed_ops)
    except RejectedEdit as rej:
        errors += rej.errors
    except Exception as exc:  # defensive: a planner bug must reject, not crash the run
        errors.append(f"internal planning error: {type(exc).__name__}: {exc}")
    if plan is None or errors:
        return ValidationResult(False, errors, proposal=proposal)
    cumulative = (ctx.tracker.count_after(model, plan) if ctx.tracker is not None else plan.n_scalars)
    if cumulative > ctx.edit_budget:
        errors.append(f"scalar budget exceeded: {cumulative} changed scalars after the edit > budget {ctx.edit_budget}")
    rel = plan.delta_sq ** 0.5 / max(_global_norm(model), 1e-30)
    return ValidationResult(not errors, errors, plan.n_scalars, rel, cumulative, proposal, plan)
