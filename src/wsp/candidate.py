"""Evaluate one proposal end to end: validate -> clone -> apply -> score on val_edit.

Shared by every editing arm: arms differ only in how proposals are chosen, never in how they are applied."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from torch import nn

from src.models.checkpoint import clone_model
from src.training.evaluate import EvalResult
from src.utils.logger import JsonlLogger
from src.wsp.edit_validator import ValidationContext, ValidationResult, validate_proposal
from src.wsp.weight_editor import EditRecord, apply_edit


class CandidateBudgetExhausted(RuntimeError):
    """evaluate_candidate was called after the evaluation budget was used up (a caller bug)."""


@dataclass
class CandidateBudget:
    """Run-level candidate-evaluation budget. Only evaluated candidates consume it;
    validator rejections are counted separately."""

    max_evaluations: int
    evaluations_used: int = 0
    validator_rejections: int = 0

    @property
    def remaining(self) -> int:
        return self.max_evaluations - self.evaluations_used


@dataclass
class CandidateResult:
    """Result for one proposal. ``model`` is the edited clone (None if the proposal was invalid)."""

    status: Literal["invalid", "evaluated"]
    validation: ValidationResult
    edit_record: EditRecord | None = None
    metrics: dict[str, Any] | None = None
    model: nn.Module | None = field(default=None, repr=False)


def evaluate_candidate(model: nn.Module, raw_proposal: Any, ctx: ValidationContext,
                       evaluate_fn: Callable[[nn.Module], EvalResult], budget: CandidateBudget,
                       run_dir: Path | None = None,
                       log: JsonlLogger | None = None) -> CandidateResult:
    """Validate ``raw_proposal`` on ``model`` (read-only); if valid, clone, apply and score on val_edit.

    Invalid proposals cost no evaluation but are logged and counted in ``budget.validator_rejections``.
    """
    validation = validate_proposal(raw_proposal, model, ctx)
    if not validation.ok:
        budget.validator_rejections += 1
        result = CandidateResult("invalid", validation)
        _log(log, result)
        return result
    if budget.remaining <= 0:
        raise CandidateBudgetExhausted(f"all {budget.max_evaluations} candidate evaluations are used")
    candidate = clone_model(model)
    record = apply_edit(candidate, validation.proposal, ctx.dsl, ctx.allowed_layers, ctx.allowed_ops, run_dir)
    scores = evaluate_fn(candidate)
    budget.evaluations_used += 1
    result = CandidateResult("evaluated", validation, record,
                             {"val_edit_loss": scores.loss, "val_edit_accuracy": scores.accuracy}, candidate)
    _log(log, result)
    return result


def _log(log: JsonlLogger | None, r: CandidateResult) -> None:
    if log is None:
        return
    p = r.validation.proposal
    log.log("candidate", status=r.status, proposal=p.model_dump() if p else None, errors=r.validation.errors,
            n_scalars=r.validation.n_scalars, rel_norm=r.validation.rel_norm,
            cumulative_after=r.validation.cumulative_after,
            checkpoint_sha256=r.edit_record.state_sha256 if r.edit_record else None,
            npz_path=r.edit_record.npz_path if r.edit_record else None, metrics=r.metrics)
