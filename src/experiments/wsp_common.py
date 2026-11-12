"""Shared setup for LLM-driven arms: evaluation functions, controller settings, agent, run reporting."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from src.data.dataset import ensure_diag_indices, get_cached_batches, get_diag_batches
from src.data.splits import load_indices
from src.training.evaluate import EvalResult, evaluate
from src.utils.config import Config, resolve_path
from src.utils.logger import JsonlLogger
from src.wsp.llm_agent import CostGuard, FakeLLM, LLMAgent, build_llm_client
from src.wsp.wsp_controller import ControllerSettings


def edit_budget_scalars(cfg: Config, fraction: float | None = None) -> int:
    """floor(fraction * total parameters); default is the configured 0.1%."""
    return math.floor(cfg.model.expected_total_params * (cfg.budgets.edit_budget_fraction if fraction is None else fraction))


class WSPData:
    """val_edit / val_select held in RAM as preprocessed batches, plus the fixed diagnostic subset."""

    def __init__(self, cfg: Config, device: torch.device) -> None:
        self.cfg, self.device = cfg, device
        ensure_diag_indices(cfg)
        self.edit_batches = get_cached_batches("val_edit", cfg)
        self.select_batches = get_cached_batches("val_select", cfg)
        self.diag_batches = get_diag_batches(cfg)
        self.val_edit_indices = load_indices(resolve_path(cfg.paths.splits_dir), "val_edit")

    def eval_edit(self, model: nn.Module) -> EvalResult:
        return evaluate(model, self.edit_batches, self.device, amp=self.cfg.training.eval_amp, num_classes=self.cfg.data.num_classes)

    def eval_select(self, model: nn.Module) -> EvalResult:
        return evaluate(model, self.select_batches, self.device, amp=self.cfg.training.eval_amp, num_classes=self.cfg.data.num_classes)


def make_settings(cfg: Config, arm: str, mode: str, allowed_layers: tuple[str, ...] | None = None) -> ControllerSettings:
    """Controller settings straight from config (budgets and acceptance rule of PROTOCOL.md Sections 8-9)."""
    b, a = cfg.budgets, cfg.acceptance
    return ControllerSettings(arm=arm, mode=mode, allowed_layers=allowed_layers or tuple(cfg.dsl.allowed_layers),  # type: ignore[arg-type]
                              allowed_ops=tuple(cfg.dsl.allowed_ops), edit_budget=edit_budget_scalars(cfg),
                              max_iterations=b.max_iterations, candidates_per_iteration=b.candidates_per_iteration,
                              max_candidates=b.max_candidates, min_loss_decrease=a.min_loss_decrease,
                              require_accuracy_non_decrease=a.require_accuracy_non_decrease,
                              stop_after_unproductive=a.stop_after_unproductive if arm not in a.disable_early_stop_for else None)


def dry_run_client() -> FakeLLM:
    """Scripted client: each call returns one tiny valid edit, one invalid edit and a no_edit proposal."""
    counter = iter(range(10**6))

    def reply(_messages: Any, _system: str) -> str:
        n = next(counter)
        base = {"reason": "dry run", "evidence": ["metrics.loss"], "expected_effect": "none", "confidence": 0.1, "risk": "low"}
        good = {**base, "proposal_id": f"dry{n}a", "operations": [{"operation": "scale_output_channel", "target_layer": "features.10",
                "channel_indices": [n % 256], "scale_factors": [1.05]}]}
        bad = {**base, "proposal_id": f"dry{n}b", "operations": [{"operation": "scale_output_channel", "target_layer": "features.10",
               "channel_indices": [0], "scale_factors": [9.0]}]}
        none = {**base, "proposal_id": f"dry{n}c", "operations": [{"operation": "no_edit", "reason": "dry run"}]}
        return json.dumps([good, bad, none])

    return FakeLLM([reply], repeat_last=True, tokens=(1000, 300))


def make_agent(cfg: Config, mode: str, settings: ControllerSettings, log: JsonlLogger, dry_run: bool) -> LLMAgent:
    """Agent on the real provider (from .env, cached, cost-capped) or on the scripted FakeLLM."""
    if dry_run:
        return LLMAgent(dry_run_client(), cfg, mode, settings.allowed_layers, settings.allowed_ops, log)  # type: ignore[arg-type]
    guard = CostGuard(cfg.llm.max_calls_per_run, cfg.llm.max_usd_per_run)
    client, _ = build_llm_client(cfg, guard)
    return LLMAgent(client, cfg, mode, settings.allowed_layers, settings.allowed_ops, log)  # type: ignore[arg-type]


def run_dirs(cfg: Config, arm: str, seed: int, sub: str = "") -> tuple[Path, Path]:
    """(results dir, checkpoint dir) for an arm and seed."""
    return (resolve_path(cfg.paths.results_dir) / arm / f"seed{seed}" / sub,
            resolve_path(cfg.paths.checkpoints_dir) / arm / f"seed{seed}" / sub)


def print_run_table(history_path: Path) -> None:
    """Per-candidate table from history.jsonl."""
    print(f"{'it':>2} {'#':>1} {'decision':<9} {'scalars':>7} {'val_edit acc':>12} {'loss':>7}  proposal")
    for line in history_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["type"] == "candidate":
            m = r["val_edit"] or {}
            acc = f"{m['val_edit_accuracy']:.4f}" if m else "-"
            loss = f"{m['val_edit_loss']:.4f}" if m else "-"
            summary = ";".join(f"{op.get('operation')}" for op in (r['proposal'].get('operations', []) if isinstance(r['proposal'], dict) else []))
            print(f"{r['iteration']:>2} {r['candidate_index']:>1} {r['decision']:<9} {r['n_scalars']:>7} {acc:>12} {loss:>7}  {summary}"
                  + (f"  ERR: {r['errors'][0][:70]}" if r["errors"] else ""))
        elif r["type"] == "llm_failure":
            print(f"{r['iteration']:>2} - llm_failure {r['error'][:80]}")
