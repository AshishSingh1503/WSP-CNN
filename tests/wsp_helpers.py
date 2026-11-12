"""Shared helpers for controller/agent tests: fake evaluation functions and scripted proposals."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.training.evaluate import EvalResult, metrics_from_logits
from src.utils.config import load_config
from src.utils.logger import JsonlLogger
from src.wsp.wsp_controller import ControllerSettings, WSPController

CFG = load_config()
BASE = {"reason": "r", "evidence": ["metrics.loss"], "expected_effect": "e", "confidence": 0.5, "risk": "low"}


def proposal(pid: str, *ops: dict[str, Any]) -> dict[str, Any]:
    return {**BASE, "proposal_id": pid, "operations": list(ops)}


def bias_op(shift: float, idx: int = 3) -> dict[str, Any]:
    return {"operation": "bias_shift", "target_layer": "classifier.6", "indices": [idx], "shifts": [shift]}


BAD_OP = {"operation": "scale_output_channel", "target_layer": "features.10", "channel_indices": [0], "scale_factors": [9.0]}
NO_EDIT = {"operation": "no_edit", "reason": "nothing clearly helps"}


def fake_result(loss: float, acc: float, n: int = 10) -> EvalResult:
    """An EvalResult with chosen loss/accuracy (other fields come from dummy logits)."""
    labels = torch.arange(n) % 10
    logits = torch.zeros(n, 10)
    logits[torch.arange(n), labels] = 3.0
    r = metrics_from_logits(logits, labels)
    r.loss, r.accuracy = loss, acc
    return r


def make_eval_fns(select_slope: float, b0: float):
    """val_edit loss falls by 1.0 per unit shift of classifier.6.bias[3] (relative to its start value b0);
    val_select accuracy moves by select_slope per unit shift."""
    def edit(model) -> EvalResult:
        return fake_result(loss=2.0 - (float(model.classifier[6].bias[3]) - b0), acc=0.80)
    def select(model) -> EvalResult:
        return fake_result(loss=1.0, acc=0.80 + select_slope * (float(model.classifier[6].bias[3]) - b0))
    return edit, select


def synthetic_diag(n: int = 10) -> list[tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator().manual_seed(0)
    return [(torch.randn(n, 3, 224, 224, generator=g), torch.arange(n) % 10)]


def make_controller(tmp: Path, model, agent, mode: str = "pure", select_slope: float = -0.05,
                    **overrides: Any) -> WSPController:
    from src.experiments.wsp_common import make_settings
    settings: ControllerSettings = replace(make_settings(CFG, "TEST", mode), **overrides)
    edit, select = make_eval_fns(select_slope, float(model.classifier[6].bias[3]))
    log = JsonlLogger(tmp / "log.jsonl", "t", 0, "h")
    return WSPController(CFG, settings, agent, model, edit, select, synthetic_diag(), np.arange(10) + 100, 0,
                         tmp / "run", tmp / "ckpt", log, "cpu")


def read_history(tmp: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (tmp / "run" / "history.jsonl").read_text().splitlines()]
