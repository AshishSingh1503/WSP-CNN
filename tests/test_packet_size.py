"""Packet JSON stays under the character budget (synthetic AlexNet always; real checkpoint if available)."""
import json

import pytest
import torch

from src.diagnostics.packet import build_hybrid_packet, build_pure_packet, to_prompt_json
from src.models.checkpoint import read_registry
from src.utils.config import load_config
from tests.test_packet_isolation import _synthetic


def test_synthetic_alexnet_under_budget() -> None:
    cfg = load_config()
    model, kwargs = _synthetic()
    for build in (build_pure_packet, build_hybrid_packet):
        text = to_prompt_json(build(model, **kwargs), cfg.diagnostics.max_chars, cfg.diagnostics.sig_figs)
        assert len(text) <= cfg.diagnostics.max_chars
        json.loads(text)


def test_truncation_keeps_protected_and_recent_history() -> None:
    from src.diagnostics.packet import EditHistoryItem
    model, kwargs = _synthetic()
    hist = [EditHistoryItem(iteration=i, proposal_summary="x" * 300, decision="rejected") for i in range(60)]
    packet = build_pure_packet(model, **{**kwargs, "edit_history": hist})
    text = to_prompt_json(packet, max_chars=22000)
    data = json.loads(text)
    assert len(text) <= 22000 and "edit_history" in "".join(data["_truncated"])
    assert data["edit_history"][-1]["iteration"] == 59           # newest kept
    assert len(data["allowed_ops"]) == 5 and len(data["allowed_layers"]) == 8  # protected
    assert len(data["confusion"]["confusion_matrix"]) == 10


def test_rounding_to_four_significant_figures() -> None:
    from src.diagnostics.packet import _round
    assert _round({"a": 0.123456789, "b": [12345.678, 7], "c": True}, 4) == {"a": 0.1235, "b": [12350.0, 7], "c": True}


def test_cannot_fit_raises() -> None:
    model, kwargs = _synthetic()
    with pytest.raises(ValueError):
        to_prompt_json(build_pure_packet(model, **kwargs), max_chars=200)


@pytest.mark.skipif("0" not in read_registry(load_config()), reason="seed-0 common checkpoint not created yet")
def test_real_checkpoint_under_budget() -> None:
    from src.data.dataset import get_diag_batches, get_loader
    from src.models.checkpoint import require_common_checkpoint
    from src.data.splits import load_indices
    from src.training.evaluate import evaluate
    from src.utils.config import resolve_path
    cfg = load_config()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = require_common_checkpoint(cfg, 0, dev)
    result = evaluate(model, get_loader("val_edit", cfg), dev)
    kwargs = dict(eval_result=result, val_edit_indices=load_indices(resolve_path(cfg.paths.splits_dir), "val_edit"),
                  diag_batches=get_diag_batches(cfg), device=dev, cfg=cfg, seed=0, iteration=1,
                  remaining_edit_budget=57044, remaining_candidate_budget=40, edit_history=[])
    for build in (build_pure_packet, build_hybrid_packet):
        text = to_prompt_json(build(model, **kwargs), cfg.diagnostics.max_chars, cfg.diagnostics.sig_figs)
        assert len(text) <= cfg.diagnostics.max_chars and "_truncated" not in text
