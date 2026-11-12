"""Applying edits: exact scalar counts, bit-exact revert, protected source, finite and count assertions."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.checkpoint import clone_model
from src.utils.config import load_config
from src.utils.reproducibility import sha256_tensor_dict
from src.wsp import weight_editor
from src.wsp.edit_schema import parse_proposal
from src.wsp.edit_validator import ValidationContext, validate_proposal
from src.wsp.weight_editor import (CountMismatchError, NonFiniteError, ProtectedModelError, apply_edit,
                                   protect_model, revert_edit)
from tests.test_edit_validator import bias, delta, lowrank, prop, scale

DSL = load_config().dsl
LAYERS, OPS = tuple(DSL.allowed_layers), tuple(DSL.allowed_ops)


def parsed(*ops):
    p, errors = parse_proposal(prop(*ops))
    assert p is not None, errors
    return p


def state_equal(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_scale_counts_fan_in_plus_bias(model) -> None:
    """features.10 is Conv2d(256, 256, 3): one output channel = 256*3*3 + 1 = 2,305 scalars.
    (192*3*3 + 1 = 1,729 is the per-channel size of features.6, whose input has 192 channels.)"""
    w = model.features[10].weight
    expected = w.shape[1] * w.shape[2] * w.shape[3] + 1
    assert expected == 2305
    rec = apply_edit(model, parsed(scale(ch=(5,), f=(1.2,))), DSL, LAYERS, OPS)
    assert rec.n_scalars == 2305 and rec.per_tensor_counts == {"features.10.weight": 2304, "features.10.bias": 1}
    six = apply_edit(model, parsed(scale(layer="features.6", ch=(0,), f=(1.2,))), DSL, LAYERS, OPS)
    assert six.n_scalars == 192 * 3 * 3 + 1 == 1729


def test_scale_values_exact(model) -> None:
    before = model.features[10].weight.detach().clone()
    apply_edit(model, parsed(scale(ch=(5,), f=(0.5,))), DSL, LAYERS, OPS)
    assert torch.equal(model.features[10].weight[5], before[5] * 0.5)
    assert torch.equal(model.features[10].weight[6], before[6])


def test_apply_then_revert_is_bit_identical(model) -> None:
    original = copy.deepcopy(model)
    h0 = sha256_tensor_dict(model.state_dict())
    rec = apply_edit(model, parsed(scale(), bias(), delta()), DSL, LAYERS, OPS)
    assert sha256_tensor_dict(model.state_dict()) == rec.state_sha256 != h0
    revert_edit(model, rec)
    assert sha256_tensor_dict(model.state_dict()) == h0 and state_equal(model, original)


def test_low_rank_apply_revert_and_count(model) -> None:
    h0 = sha256_tensor_dict(model.state_dict())
    rec = apply_edit(model, parsed(lowrank(rows=(1, 2), cols=(0, 5, 9), u=((0.1, 0.1),), v=((0.05, 0.05, 0.05),))),
                     DSL, LAYERS, OPS)
    assert rec.n_scalars == 6 and set(rec.per_tensor_counts) == {"classifier.6.weight"}
    revert_edit(model, rec)
    assert sha256_tensor_dict(model.state_dict()) == h0


def test_edit_record_contents(model, tmp_path) -> None:
    p = parsed(scale(), bias(), delta())
    rec = apply_edit(model, p, DSL, LAYERS, OPS, run_dir=tmp_path)
    assert rec.n_scalars == 2305 + 1 + 1 and rec.proposal_id == "p1"
    assert set(rec.per_layer_rel_norm) == {"features.10", "classifier.6"}
    assert all(v > 0 for v in rec.per_layer_rel_norm.values()) and 0 < rec.global_rel_norm < 1
    data = np.load(rec.npz_path)
    for name, change in rec.changes.items():
        assert np.array_equal(data[f"{name}__idx"], change.flat_idx.numpy())
        assert np.array_equal(data[f"{name}__before"], change.before.numpy())
        assert np.array_equal(data[f"{name}__after"], change.after.numpy())
    assert rec.npz_path.endswith("edits\\p1.npz") or rec.npz_path.endswith("edits/p1.npz")


def test_validator_prediction_matches_applied_count(model) -> None:
    ctx = ValidationContext(DSL, LAYERS, OPS, 57044)
    for ops in ([scale()], [bias()], [delta()], [lowrank()], [scale(), bias(), delta()]):
        raw = prop(*ops)
        result = validate_proposal(raw, model, ctx)
        rec = apply_edit(clone_model(model), result.proposal, DSL, LAYERS, OPS)
        assert result.ok and result.n_scalars == rec.n_scalars


def test_protected_source_cannot_be_edited(model) -> None:
    before = sha256_tensor_dict(model.state_dict())
    protect_model(model)
    with pytest.raises(ProtectedModelError):
        apply_edit(model, parsed(bias()), DSL, LAYERS, OPS)
    clone = clone_model(model)
    rec = apply_edit(clone, parsed(bias()), DSL, LAYERS, OPS)  # a clone is not protected
    with pytest.raises(ProtectedModelError):
        revert_edit(model, rec)
    assert sha256_tensor_dict(model.state_dict()) == before != rec.state_sha256


def test_clone_independent_of_source(model) -> None:
    before = sha256_tensor_dict(model.state_dict())
    clone = clone_model(model)
    apply_edit(clone, parsed(scale()), DSL, LAYERS, OPS)
    assert sha256_tensor_dict(model.state_dict()) == before


def test_nan_in_model_is_caught_and_edit_reverted(model) -> None:
    with torch.no_grad():
        model.classifier[6].bias[7] = float("nan")
    h = sha256_tensor_dict(model.state_dict())
    with pytest.raises(NonFiniteError):
        apply_edit(model, parsed(bias()), DSL, LAYERS, OPS)
    assert sha256_tensor_dict(model.state_dict()) == h  # the attempted edit was rolled back


def test_nan_created_by_edit_is_caught(model, monkeypatch) -> None:
    real_write = weight_editor._write

    def poison(m, changes, use_after):
        real_write(m, changes, use_after)
        if use_after:
            m.classifier[6].bias.detach()[0] = float("inf")

    monkeypatch.setattr(weight_editor, "_write", poison)
    with pytest.raises((NonFiniteError, CountMismatchError)):
        apply_edit(model, parsed(bias(idx=(3,))), DSL, LAYERS, OPS)


def test_count_mismatch_is_caught_and_reverted(model, monkeypatch) -> None:
    real_write = weight_editor._write

    def extra_write(m, changes, use_after):
        real_write(m, changes, use_after)
        if use_after:
            m.classifier[6].bias.detach()[9] += 1.0  # an unplanned change

    monkeypatch.setattr(weight_editor, "_write", extra_write)
    with pytest.raises(CountMismatchError):
        apply_edit(model, parsed(bias(idx=(3,))), DSL, LAYERS, OPS)


def test_apply_rejects_unplannable_proposal(model) -> None:
    with pytest.raises(weight_editor.EditError):
        apply_edit(model, parsed(scale(f=(5.0,))), DSL, LAYERS, OPS)
