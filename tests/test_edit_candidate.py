"""evaluate_candidate: validator rejections are free and counted; evaluations are budgeted; source is untouched."""
import pytest
import torch

from src.training.evaluate import metrics_from_logits
from src.utils.config import load_config
from src.utils.reproducibility import sha256_tensor_dict
from src.wsp.candidate import CandidateBudget, CandidateBudgetExhausted, evaluate_candidate
from src.wsp.edit_validator import ValidationContext
from src.wsp.weight_editor import protect_model
from tests.test_edit_validator import bias, prop, scale

DSL = load_config().dsl
CTX = ValidationContext(DSL, tuple(DSL.allowed_layers), tuple(DSL.allowed_ops), 57044)
X = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(0))
Y = torch.tensor([1, 2])


def score(m) -> object:
    with torch.no_grad():
        return metrics_from_logits(m.eval()(X), Y)


def test_invalid_costs_no_evaluation(model) -> None:
    budget = CandidateBudget(max_evaluations=2)
    calls = []
    res = evaluate_candidate(model, prop(scale(f=(9.0,))), CTX, lambda m: calls.append(1) or score(m), budget)
    assert res.status == "invalid" and res.model is None and not calls
    assert budget.evaluations_used == 0 and budget.validator_rejections == 1 and budget.remaining == 2


def test_valid_candidate_consumes_one_evaluation(model, tmp_path) -> None:
    budget = CandidateBudget(max_evaluations=2)
    h = sha256_tensor_dict(model.state_dict())
    protect_model(model)
    res = evaluate_candidate(model, prop(bias()), CTX, score, budget, run_dir=tmp_path)
    assert res.status == "evaluated" and budget.evaluations_used == 1 and budget.validator_rejections == 0
    assert set(res.metrics) == {"val_edit_loss", "val_edit_accuracy"}
    assert sha256_tensor_dict(model.state_dict()) == h                      # source untouched
    assert sha256_tensor_dict(res.model.state_dict()) == res.edit_record.state_sha256 != h
    assert res.model is not model


def test_evaluation_budget_is_enforced(model) -> None:
    budget = CandidateBudget(max_evaluations=1)
    evaluate_candidate(model, prop(bias()), CTX, score, budget)
    with pytest.raises(CandidateBudgetExhausted):
        evaluate_candidate(model, prop(bias(idx=(4,))), CTX, score, budget)
    evaluate_candidate(model, prop(scale(f=(9.0,))), CTX, score, budget)  # invalid proposals still just logged
    assert budget.validator_rejections == 1


def test_candidate_logged(model, tmp_path) -> None:
    from src.utils.logger import JsonlLogger
    with JsonlLogger(tmp_path / "log.jsonl", "t", 0, "h") as log:
        evaluate_candidate(model, prop(bias()), CTX, score, CandidateBudget(1), log=log)
        evaluate_candidate(model, "{bad", CTX, score, CandidateBudget(1), log=log)
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert len(lines) == 2 and '"status": "evaluated"' in lines[0] and '"status": "invalid"' in lines[1]
    assert "checkpoint_sha256" in lines[0] and "errors" in lines[1]
