"""Cumulative edit-budget accounting: always recomputed as a diff against the common checkpoint."""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from src.wsp.bitwise import differs
from src.wsp.edit_plan import EditPlan


def reference_params(model: nn.Module) -> dict[str, torch.Tensor]:
    """CPU copy of every parameter (the common checkpoint's values)."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()}


class BudgetTracker:
    """Counts scalars that differ (bitwise) from the reference checkpoint; the budget is a scalar count."""

    def __init__(self, reference: Mapping[str, torch.Tensor], budget: int) -> None:
        self.reference, self.budget = dict(reference), budget

    def per_tensor(self, model: nn.Module) -> dict[str, int]:
        """Exact changed-scalar count per parameter tensor, recomputed from scratch."""
        return {n: int(differs(p.detach().cpu(), self.reference[n]).sum()) for n, p in model.named_parameters()}

    def count(self, model: nn.Module) -> int:
        """Total changed scalars of ``model`` vs the reference."""
        return sum(self.per_tensor(model).values())

    def count_after(self, model: nn.Module, plan: EditPlan) -> int:
        """Changed-scalar count (vs reference) the model WOULD have after applying ``plan``."""
        counts = self.per_tensor(model)
        for name, change in plan.changes.items():
            t = dict(model.named_parameters())[name].detach().cpu().clone().reshape(-1)
            t[change.flat_idx] = change.after
            counts[name] = int(differs(t, self.reference[name].reshape(-1)).sum())
        return sum(counts.values())

    def remaining(self, model: nn.Module) -> int:
        """Budget left for ``model`` (never negative)."""
        return max(0, self.budget - self.count(model))
