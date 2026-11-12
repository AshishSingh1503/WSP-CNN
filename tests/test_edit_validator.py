"""Every malformed or out-of-policy proposal is rejected with a clear error; valid ones are accepted."""
import copy
import json

import pytest
import yaml

from src.utils.config import DEFAULT_CONFIG_PATH, Config, load_config
from src.wsp.budget import BudgetTracker, reference_params
from src.wsp.edit_validator import ValidationContext, validate_proposal

CFG = load_config()
BASE = {"proposal_id": "p1", "reason": "r", "evidence": ["metrics.loss"], "expected_effect": "e",
        "confidence": 0.5, "risk": "low"}


def ctx(cfg: Config = CFG, layers=None, ops=None, budget: int = 57044, tracker=None) -> ValidationContext:
    return ValidationContext(cfg.dsl, tuple(layers or cfg.dsl.allowed_layers), tuple(ops or cfg.dsl.allowed_ops),
                             budget, tracker)


def prop(*operations, **overrides) -> dict:
    return {**BASE, "operations": list(operations), **overrides}


def scale(layer="features.10", ch=(3,), f=(1.2,)):
    return {"operation": "scale_output_channel", "target_layer": layer, "channel_indices": list(ch), "scale_factors": list(f)}


def bias(layer="classifier.6", idx=(3,), s=(0.5,)):
    return {"operation": "bias_shift", "target_layer": layer, "indices": list(idx), "shifts": list(s)}


def delta(param="classifier.6.weight", idx=((3, 100),), d=(0.01,)):
    return {"operation": "add_delta", "target_param": param, "indices": [list(i) for i in idx], "deltas": list(d)}


def lowrank(rows=(3,), cols=(0, 1, 2, 3), u=((0.1,),), v=((0.05,) * 4,), param="classifier.6.weight"):
    return {"operation": "low_rank_update", "target_param": param, "rows": list(rows), "cols": list(cols),
            "u": [list(x) for x in u], "v": [list(x) for x in v]}


VALID = [prop(scale()), prop(bias()), prop(delta()), prop(lowrank()),
         prop(scale(), bias(), delta()), prop({"operation": "no_edit", "reason": "none"})]


@pytest.mark.parametrize("proposal", VALID)
def test_valid_proposals_accepted(model, proposal) -> None:
    result = validate_proposal(proposal, model, ctx())
    assert result.ok, result.errors


REJECTED = [
    ("malformed json", "{not json", "malformed JSON"),
    ("json list", "[]", "must be a JSON object"),
    ("json null", "null", "must be a JSON object"),
    ("unknown operation", prop({"operation": "rotate_filters", "target_layer": "features.0"}), "operations"),
    ("extra top-level field", {**prop(scale()), "secret": 1}, "secret"),
    ("extra op field", prop({**scale(), "extra": 1}), "extra"),
    ("missing field", {k: v for k, v in prop(scale()).items() if k != "risk"}, "risk"),
    ("unknown layer", prop(scale(layer="features.99")), "unknown layer"),
    ("unknown param", prop(delta(param="features.10.gamma")), "unknown parameter"),
    ("non-conv layer", prop(scale(layer="features.1")), "unknown layer"),
    ("index out of range", prop(delta(idx=((999, 0),))), "out of range"),
    ("wrong index arity", prop(delta(idx=((3,),))), "coordinates"),
    ("channel out of range", prop(scale(ch=(999,))), "out of range"),
    ("bias index out of range", prop(bias(idx=(99,))), "out of range"),
    ("negative index", prop(delta(idx=((-1, 0),))), "out of range"),
    ("duplicate indices", prop(delta(idx=((3, 1), (3, 1)), d=(0.01, 0.01))), "duplicate"),
    ("duplicate channels", prop(scale(ch=(3, 3), f=(1.1, 1.1))), "duplicate"),
    ("delta too large", prop(delta(d=(0.5,))), "exceeds"),
    ("scale too large", prop(scale(f=(2.0,))), "outside"),
    ("scale too small", prop(scale(f=(0.1,))), "outside"),
    ("zero scale not allowed", prop(scale(f=(0.0,))), "outside"),
    ("bias shift too large elsewhere", prop(bias(layer="features.0", idx=(0,), s=(0.5,))), "exceeds"),
    ("bias shift too large head", prop(bias(s=(1.5,))), "exceeds"),
    ("rank too high", prop(lowrank(u=((0.1,),) * 5, v=((0.05,) * 4,) * 5)), "rank"),
    ("low-rank norm too big", prop(lowrank(u=((5.0,),), v=((5.0,) * 4,))), "||update||"),
    ("low-rank on conv", prop(lowrank(param="features.10.weight")), "unknown layer"),
    ("low-rank on bias", prop(lowrank(param="classifier.6.bias")), "not a known weight"),
    ("too many ops", prop(bias(idx=(0,)), bias(idx=(1,)), bias(idx=(2,)), bias(idx=(3,))), "operations"),
    ("no_edit mixed", prop(bias(), {"operation": "no_edit", "reason": ""}), "only operation"),
    ("empty operations", prop(), "operations"),
    ("long reason", prop(bias(), reason="x" * 801), "reason"),
    ("bad id", prop(bias(), proposal_id="a/b"), "proposal_id"),
    ("confidence out of range", prop(bias(), confidence=1.5), "confidence"),
    ("no-op edit", prop(scale(f=(1.0,))), "no parameter value"),
    ("numeric string", prop(bias(s=("0.5",))), "valid number"),
    ("float index", prop(bias(idx=(3.0,))), "valid integer"),
]


@pytest.mark.parametrize("name,proposal,fragment", REJECTED, ids=[r[0] for r in REJECTED])
def test_rejected_with_clear_error(model, name, proposal, fragment) -> None:
    result = validate_proposal(proposal, model, ctx())
    assert not result.ok and result.errors, name
    assert any(fragment in e for e in result.errors), result.errors


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nan_inf_rejected(model, token) -> None:
    text = json.dumps(prop(bias())).replace("0.5", token)
    result = validate_proposal(text, model, ctx())
    assert not result.ok and any("valid number" in e or "finite" in e for e in result.errors), result.errors


def test_layer_not_allowed(model) -> None:
    result = validate_proposal(prop(scale()), model, ctx(layers=["classifier.6"]))
    assert not result.ok and any("not allowed" in e for e in result.errors)


def test_operation_not_permitted(model) -> None:
    result = validate_proposal(prop(delta()), model, ctx(ops=["bias_shift", "no_edit"]))
    assert not result.ok and any("not permitted" in e for e in result.errors)


def test_budget_overflow(model) -> None:
    result = validate_proposal(prop(scale()), model, ctx(budget=100))  # features.10 channel = 2,305 scalars
    assert not result.ok and result.n_scalars == 2305 and any("budget" in e for e in result.errors)


def test_cumulative_budget_via_tracker(model) -> None:
    tracker = BudgetTracker(reference_params(model), budget=2400)
    assert validate_proposal(prop(scale()), model, ctx(budget=2400, tracker=tracker)).ok
    two = prop(scale(ch=(3, 4), f=(1.2, 1.2)))
    assert not validate_proposal(two, model, ctx(budget=2400, tracker=tracker)).ok


def test_zero_scale_allowed_only_by_config(model) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["dsl"]["scale_output_channel"]["allow_zero_channel"] = True
    cfg = Config.model_validate(raw)
    assert validate_proposal(prop(scale(f=(0.0,))), model, ctx(cfg)).ok
    assert not validate_proposal(prop(scale(f=(0.2,))), model, ctx(cfg)).ok  # still outside [0.5, 1.5]


def test_head_bias_limit_is_larger_than_elsewhere(model) -> None:
    assert validate_proposal(prop(bias(s=(0.9,))), model, ctx()).ok
    assert not validate_proposal(prop(bias(layer="features.0", idx=(0,), s=(0.9,))), model, ctx()).ok


@pytest.mark.parametrize("junk", [None, 5, [], b"\xff\xfe", "", "{}", {"operations": 3}, object()])
def test_never_raises_on_junk(model, junk) -> None:
    result = validate_proposal(junk, model, ctx())
    assert result.ok is False and result.errors


def test_validation_does_not_modify_model(model) -> None:
    before = {k: v.clone() for k, v in model.state_dict().items()}
    validate_proposal(prop(scale(), bias(), delta()), model, ctx())
    assert all(bool((before[k] == v).all()) for k, v in model.state_dict().items())


def test_proposal_dict_not_mutated(model) -> None:
    p = prop(scale())
    snapshot = copy.deepcopy(p)
    validate_proposal(p, model, ctx())
    assert p == snapshot
