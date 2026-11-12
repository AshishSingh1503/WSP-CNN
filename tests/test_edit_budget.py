"""Cumulative budget tracking across overlapping accepted edits (recomputed, not summed)."""
import numpy as np
import torch

from src.utils.config import load_config
from src.wsp.budget import BudgetTracker, reference_params
from src.wsp.bitwise import differs
from src.wsp.edit_schema import parse_proposal
from src.wsp.edit_validator import ValidationContext, validate_proposal
from src.wsp.weight_editor import apply_edit
from tests.test_edit_validator import bias, delta, prop, scale

DSL = load_config().dsl
LAYERS, OPS = tuple(DSL.allowed_layers), tuple(DSL.allowed_ops)


def parsed(*ops):
    return parse_proposal(prop(*ops))[0]


def independent_count(model, reference) -> int:
    """Count changed scalars with plain numpy bit comparison (independent of BudgetTracker)."""
    total = 0
    for name, p in model.named_parameters():
        a = p.detach().cpu().numpy().view(np.uint32)
        b = reference[name].numpy().view(np.uint32)
        total += int((a != b).sum())
    return total


def test_three_overlapping_edits(model) -> None:
    reference = reference_params(model)
    tracker = BudgetTracker(reference, budget=57044)
    assert tracker.count(model) == 0
    rows = [(3, i) for i in range(10)]
    e1 = parsed(delta(idx=rows, d=[0.01] * 10))                                    # 10 scalars
    e2 = parsed(delta(idx=rows[5:] + [(4, i) for i in range(5)], d=[0.01] * 10))   # 5 overlap + 5 new
    e3 = parsed(scale(layer="classifier.6", ch=(3,), f=(1.1,)), bias(idx=(3,), s=(0.1,)))  # row 3 (4096) + bias
    expected_counts = [10, 15, 15 - 10 + 4096 + 1]
    for edit, expected in zip((e1, e2, e3), expected_counts):
        predicted = tracker.count_after(model, apply_probe(model, edit))
        apply_edit(model, edit, DSL, LAYERS, OPS)
        assert predicted == expected == tracker.count(model) == independent_count(model, reference)
    assert tracker.remaining(model) == 57044 - 4102


def apply_probe(model, edit):
    from src.wsp.edit_plan import plan_edit
    return plan_edit(model, edit, DSL, LAYERS, OPS)


def test_overlap_counts_each_scalar_once(model) -> None:
    tracker = BudgetTracker(reference_params(model), budget=57044)
    same = parsed(delta(idx=[(1, 1)], d=[0.01]))
    apply_edit(model, same, DSL, LAYERS, OPS)
    first = tracker.count(model)
    apply_edit(model, parsed(delta(idx=[(1, 1)], d=[0.005])), DSL, LAYERS, OPS)
    assert first == 1 and tracker.count(model) == 1


def test_budget_enforced_by_validator_on_cumulative_count(model) -> None:
    tracker = BudgetTracker(reference_params(model), budget=15)
    ctx = ValidationContext(DSL, LAYERS, OPS, 15, tracker)
    first = prop(delta(idx=[(3, i) for i in range(10)], d=[0.01] * 10))
    r1 = validate_proposal(first, model, ctx)
    assert r1.ok and r1.cumulative_after == 10
    apply_edit(model, r1.proposal, DSL, LAYERS, OPS)
    overlap = prop(delta(idx=[(3, i) for i in range(5, 15)], d=[0.01] * 10))   # 5 new -> 15 total: fits
    assert validate_proposal(overlap, model, ctx).cumulative_after == 15
    too_many = prop(delta(idx=[(4, i) for i in range(6)], d=[0.01] * 6))        # 6 new -> 16 total: rejected
    r3 = validate_proposal(too_many, model, ctx)
    assert not r3.ok and r3.cumulative_after == 16 and any("budget" in e for e in r3.errors)


def test_bitwise_differs_is_nan_safe() -> None:
    a = torch.tensor([1.0, float("nan"), 0.0, 2.0])
    b = torch.tensor([1.0, float("nan"), -0.0, 3.0])
    assert differs(a, b).tolist() == [False, False, True, True]
