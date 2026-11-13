"""Diagnostic packets sent to the LLM. PurePacket structurally has no place for gradient information.

``build_pure_packet`` never touches autograd. The gradient computation module is imported lazily inside
``build_hybrid_packet`` only; ``gradient_schema`` (plain data classes) is needed for HybridPacket's type.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any, Iterable, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

from src.diagnostics.activation_analyzer import analyze_activations, summarize_activations
from src.diagnostics.error_analyzer import analyze_errors
from src.diagnostics.gradient_schema import GradientDiagnostics
from src.diagnostics.weight_inspector import inspect_weights
from src.training.evaluate import EvalResult
from src.utils.config import Config

LOG = logging.getLogger("wsp.packet")
PROTECTED_LISTS = {"allowed_layers", "allowed_ops", "per_class_accuracy", "confusion_matrix"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetricsSummary(_Strict):
    n: int
    loss: float
    accuracy: float
    top5_accuracy: float
    macro_f1: float
    per_class_accuracy: list[float]


class EditHistoryItem(_Strict):
    iteration: int
    proposal_summary: str
    decision: Literal["accepted", "rejected", "invalid", "no_edit"]
    val_edit_loss_change: float | None = None
    val_edit_accuracy_change: float | None = None


class _PacketBase(_Strict):
    """Fields shared by both packets (no gradient field here)."""

    model_name: str
    seed: int
    iteration: int
    metrics: MetricsSummary
    confusion: dict[str, Any]
    weight_summary: dict[str, Any]
    activation_summary: dict[str, Any]
    allowed_layers: list[str]
    allowed_ops: list[str]
    remaining_edit_budget: int
    remaining_candidate_budget: int
    edit_history: list[EditHistoryItem]


class PurePacket(_PacketBase):
    """Pure WSP input. extra='forbid' (inherited) rejects any gradient_diagnostics field."""


class HybridPacket(_PacketBase):
    """Hybrid WSP input = the Pure fields + gradient diagnostics."""

    gradient_diagnostics: GradientDiagnostics


def _pure_fields(model: nn.Module, *, eval_result: EvalResult, val_edit_indices: np.ndarray,
                 diag_batches: Iterable[tuple[torch.Tensor, torch.Tensor]], device: str | torch.device,
                 cfg: Config, seed: int, iteration: int, remaining_edit_budget: int,
                 remaining_candidate_budget: int, edit_history: list[EditHistoryItem]) -> dict[str, Any]:
    d = cfg.diagnostics
    metrics = MetricsSummary(n=eval_result.n, loss=eval_result.loss, accuracy=eval_result.accuracy,
                             top5_accuracy=eval_result.top5_accuracy, macro_f1=eval_result.macro_f1,
                             per_class_accuracy=eval_result.per_class_accuracy)
    activation_summary = (summarize_activations(analyze_activations(model, diag_batches, device, d, cfg.data.num_classes), d)
                          if d.include_activation_stats else {})
    return {"model_name": d.model_name, "seed": seed, "iteration": iteration, "metrics": metrics,
            "confusion": analyze_errors(eval_result, val_edit_indices, d) if d.include_confusion else {},
            "weight_summary": inspect_weights(model, d.weight_topk) if d.include_weight_stats else {},
            "activation_summary": activation_summary,
            "allowed_layers": list(cfg.dsl.allowed_layers), "allowed_ops": list(cfg.dsl.allowed_ops),
            "remaining_edit_budget": remaining_edit_budget,
            "remaining_candidate_budget": remaining_candidate_budget, "edit_history": edit_history}


def build_pure_packet(model: nn.Module, **kwargs: Any) -> PurePacket:
    """Packet from weights, forward activations and val_edit errors only (no gradients)."""
    kwargs["diag_batches"] = list(kwargs["diag_batches"])
    with torch.no_grad():
        return PurePacket(**_pure_fields(model, **kwargs))


def build_hybrid_packet(model: nn.Module, **kwargs: Any) -> HybridPacket:
    """Pure packet fields plus gradient diagnostics (Hybrid WSP only)."""
    from src.diagnostics.gradient_analyzer import analyze_gradients  # lazy: keeps Pure paths gradient-free
    kwargs["diag_batches"] = list(kwargs["diag_batches"])  # iterated twice
    with torch.no_grad():
        fields = _pure_fields(model, **kwargs)
    grads = analyze_gradients(model, kwargs["diag_batches"], kwargs["device"], kwargs["cfg"].diagnostics)
    return HybridPacket(**fields, gradient_diagnostics=grads)


# --- prompt serialisation ---

def _round(obj: Any, sig: int) -> Any:
    if isinstance(obj, bool) or isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if obj == 0 or not math.isfinite(obj) else float(f"{obj:.{sig}g}")
    if isinstance(obj, dict):
        return {k: _round(v, sig) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, sig) for v in obj]
    return obj


def _dump(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"), sort_keys=False)


def _list_paths(obj: Any, path: tuple = ()) -> list[tuple]:
    """Paths to every truncatable list (not protected, more than one element)."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found += _list_paths(v, path + (k,))
    elif isinstance(obj, list):
        if len(obj) > 1 and (not path or path[-1] not in PROTECTED_LISTS):
            found.append(path)
        for i, v in enumerate(obj):
            found += _list_paths(v, path + (i,))
    return found


def _get(obj: Any, path: tuple) -> Any:
    for p in path:
        obj = obj[p]
    return obj


def to_prompt_json(packet: _PacketBase, max_chars: int = 25000, sig_figs: int = 4,
                   log: Any | None = None) -> str:
    """Compact JSON with floats rounded to ``sig_figs``; the longest lists are shortened until it fits.

    ``edit_history`` keeps its most recent entries; other lists keep their leading (most important)
    entries. Truncated paths are listed under ``_truncated``. Logs chars and estimated tokens (chars/4).
    """
    data = _round(packet.model_dump(mode="json"), sig_figs)
    truncated: list[str] = []
    text = _dump(data)
    for _ in range(10000):
        if len(text) <= max_chars:
            break
        paths = _list_paths(data)
        if not paths:
            raise ValueError(f"packet cannot be shortened below {max_chars} chars (now {len(text)})")
        path = max(paths, key=lambda p: (len(_dump(_get(data, p))), p))
        lst = _get(data, path)
        keep = max(1, len(lst) * 3 // 4)
        lst[:] = lst[-keep:] if path[-1] == "edit_history" else lst[:keep]
        truncated.append("/".join(map(str, path)))
        text = _dump({**data, "_truncated": sorted(set(truncated))})
    if truncated:
        data["_truncated"] = sorted(set(truncated))
    text = _dump(data)
    msg = {"chars": len(text), "tokens_estimate": len(text) // 4, "truncated_paths": len(set(truncated))}
    LOG.info("prompt packet size %s", msg)
    if log is not None:
        log.log("prompt_packet_size", **msg)
    return text
