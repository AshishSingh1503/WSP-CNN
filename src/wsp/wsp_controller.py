"""One WSP controller for E2 (pure), E3 (hybrid), E6 (last layer) and the positive control (PC)."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from torch import nn

from src.diagnostics.packet import EditHistoryItem, build_hybrid_packet, build_pure_packet, to_prompt_json
from src.models.checkpoint import clone_model
from src.models.delta import save_delta
from src.training.evaluate import EvalResult
from src.utils.config import Config, config_hash
from src.utils.logger import JsonlLogger
from src.utils.param_diff import diff_stats
from src.utils.reproducibility import sha256_tensor_dict
from src.wsp.budget import BudgetTracker, reference_params
from src.wsp.candidate import (CandidateBudget, CandidateBudgetExhausted, CandidateResult, evaluate_candidate)
from src.wsp.edit_validator import ValidationContext
from src.wsp.llm_agent import AgentResult, BudgetExceeded, LLMAgent
from src.wsp.weight_editor import protect_model

SUMMARY_CHARS = 200
DELTA_NAME = "final_delta.npz"


@dataclass(frozen=True)
class ControllerSettings:
    """Everything that differs between arms; all values come from config."""

    arm: str
    mode: Literal["pure", "hybrid"]
    allowed_layers: tuple[str, ...]
    allowed_ops: tuple[str, ...]
    edit_budget: int
    max_iterations: int
    candidates_per_iteration: int
    max_candidates: int
    min_loss_decrease: float
    require_accuracy_non_decrease: bool
    stop_after_unproductive: int | None


@dataclass
class StateRecord:
    """An accepted state (index 0 is the start state)."""

    index: int
    iteration: int
    proposal_id: str | None
    val_edit: dict[str, float]
    val_select: dict[str, float]
    changed_scalars: int


@dataclass
class RunResult:
    arm: str
    seed: int
    stop_reason: str
    states: list[StateRecord]
    final_index: int
    final_model: nn.Module = field(repr=False)
    diff: dict[str, Any]
    counts: dict[str, int]
    cost: dict[str, Any]
    checkpoint_sha256: str
    run_dir: Path


def summarise(raw: Any) -> str:
    """Short, safe text summary of a raw proposal for the history shown to the LLM."""
    try:
        ops = raw["operations"]
        parts = []
        for op in ops:
            target = op.get("target_layer") or op.get("target_param") or ""
            idx = op.get("channel_indices") or op.get("indices") or op.get("rows") or []
            parts.append(f"{op.get('operation')} {target} {json.dumps(idx)[:60]}")
        return "; ".join(parts)[:SUMMARY_CHARS]
    except Exception:  # noqa: BLE001 - arbitrary LLM output must not break bookkeeping
        return str(raw)[:SUMMARY_CHARS]


def _metrics(r: EvalResult) -> dict[str, float]:
    return {"loss": r.loss, "accuracy": r.accuracy, "macro_f1": r.macro_f1}


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class WSPController:
    """Propose -> validate -> evaluate -> accept loop shared by all LLM-driven arms."""

    def __init__(self, cfg: Config, settings: ControllerSettings, agent: LLMAgent | None,
                 start_model: nn.Module, eval_edit: Callable[[nn.Module], EvalResult],
                 eval_select: Callable[[nn.Module], EvalResult], diag_batches: list[Any],
                 val_edit_indices: np.ndarray, seed: int, run_dir: Path, ckpt_dir: Path,
                 log: JsonlLogger, device: torch.device | str, tags: dict[str, Any] | None = None) -> None:
        self.cfg, self.s, self.agent, self.start = cfg, settings, agent, protect_model(start_model)
        self.eval_edit, self.eval_select, self.diag = eval_edit, eval_select, diag_batches
        self.val_edit_indices, self.seed, self.run_dir, self.ckpt_dir = val_edit_indices, seed, run_dir, ckpt_dir
        self.log, self.device, self.tags = log, device, tags or {}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.run_dir / "history.jsonl"

    # -- audit trail --
    def _write(self, record: dict[str, Any]) -> None:
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def _packet_text(self, model: nn.Module, current: EvalResult, iteration: int, remaining_edit: int,
                     remaining_slots: int, history: list[EditHistoryItem]) -> str:
        build = build_pure_packet if self.s.mode == "pure" else build_hybrid_packet
        packet = build(model, eval_result=current, val_edit_indices=self.val_edit_indices, diag_batches=self.diag,
                       device=self.device, cfg=self.cfg, seed=self.seed, iteration=iteration,
                       remaining_edit_budget=remaining_edit, remaining_candidate_budget=remaining_slots,
                       edit_history=history)
        return to_prompt_json(packet, self.cfg.diagnostics.max_chars, self.cfg.diagnostics.sig_figs, self.log)

    def _eligible(self, cur: EvalResult, res: CandidateResult) -> bool:
        m = res.metrics
        ok = cur.loss - m["val_edit_loss"] >= self.s.min_loss_decrease
        if self.s.require_accuracy_non_decrease:
            ok = ok and m["val_edit_accuracy"] >= cur.accuracy
        return bool(ok)

    # -- main loop --
    def run(self) -> RunResult:
        s, cfg = self.s, self.cfg
        if (self.ckpt_dir / DELTA_NAME).exists():
            raise FileExistsError(f"{self.ckpt_dir / DELTA_NAME} exists; refusing to overwrite a finished run")
        if self.history_path.exists():
            raise FileExistsError(f"{self.history_path} exists; use a fresh run directory")
        wall0 = time.perf_counter()
        cur, cur_eval = self.start, self.eval_edit(self.start)
        sel0 = self.eval_select(self.start)
        tracker = BudgetTracker(reference_params(self.start), s.edit_budget)
        ctx = ValidationContext(cfg.dsl, s.allowed_layers, s.allowed_ops, s.edit_budget, tracker)
        self.start_sha256 = sha256_tensor_dict(self.start.state_dict())
        states = [StateRecord(0, 0, None, _metrics(cur_eval), _metrics(sel0), 0)]
        best_index, best_state = 0, _cpu_state(self.start)
        cand_budget, history = CandidateBudget(s.max_candidates), []
        counts = {"slots_used": 0, "iterations": 0, "accepted": 0, "failed_llm_outputs": 0}
        unproductive, stop_reason = 0, "iterations_exhausted"
        self._write({"type": "start", "arm": s.arm, "seed": self.seed, "val_edit": states[0].val_edit,
                     "val_select": states[0].val_select, "settings": asdict(s)})
        for it in range(1, s.max_iterations + 1):
            if counts["slots_used"] >= s.max_candidates:
                stop_reason = "candidate_budget"
                break
            counts["iterations"] = it
            remaining_edit = tracker.remaining(cur)
            remaining_slots = s.max_candidates - counts["slots_used"]
            text = (self._packet_text(cur, cur_eval, it, remaining_edit, remaining_slots, history)
                    if getattr(self.agent, "needs_packet", True) else "")
            ctx_info = {"remaining_edit_budget": remaining_edit, "remaining_slots": remaining_slots}
            try:
                agent_res = self.agent.propose(text, cur, it, ctx_info)
            except BudgetExceeded as err:
                stop_reason = f"cost_cap: {err}"
                self._write({"type": "stop", "iteration": it, "reason": stop_reason})
                break
            except Exception as err:  # noqa: BLE001 - provider failure must not crash the run
                agent_res = AgentResult(failed=[{"error": f"agent error: {type(err).__name__}: {err}", "raw": ""}])
            counts["failed_llm_outputs"] += len(agent_res.failed)
            for bad in agent_res.failed:
                history.append(EditHistoryItem(iteration=it, proposal_summary="LLM output unusable", decision="invalid"))
                self._write({"type": "llm_failure", "iteration": it, **bad})
            accepted_here = self._run_iteration(it, agent_res, cur, cur_eval, ctx, cand_budget, counts, history)
            self._write({"type": "iteration", "iteration": it, "packet_chars": len(text),
                         "usage": [asdict(u) for u in agent_res.usages], "inspections": agent_res.inspections,
                         "extra_proposals_dropped": agent_res.extra_dropped, "accepted": accepted_here is not None})
            if accepted_here is None:
                unproductive += 1
                if s.stop_after_unproductive and unproductive >= s.stop_after_unproductive:
                    stop_reason = f"unproductive_x{unproductive}"
                    break
                continue
            unproductive = 0
            cur, cur_eval, res = accepted_here
            protect_model(cur)
            sel = self.eval_select(cur)
            counts["accepted"] += 1
            states.append(StateRecord(len(states), it, res.edit_record.proposal_id, _metrics(cur_eval),
                                      _metrics(sel), tracker.count(cur)))
            self._write({"type": "accepted_state", "iteration": it, "state_index": states[-1].index,
                         "val_select": states[-1].val_select, "val_edit": states[-1].val_edit,
                         "checkpoint_sha256": res.edit_record.state_sha256})
            if sel.accuracy > states[best_index].val_select["accuracy"]:
                best_index, best_state = states[-1].index, _cpu_state(cur)
        else:
            stop_reason = "iterations_exhausted"
        return self._finish(states, best_index, best_state, stop_reason, counts, cand_budget, wall0)

    def _run_iteration(self, it: int, agent_res: AgentResult, cur: nn.Module, cur_eval: EvalResult,
                       ctx: ValidationContext, cand_budget: CandidateBudget, counts: dict[str, int],
                       history: list[EditHistoryItem]) -> tuple[nn.Module, EvalResult, CandidateResult] | None:
        slots_left = self.s.max_candidates - counts["slots_used"]
        results: list[tuple[int, Any, CandidateResult | None, str | None]] = []
        for j, raw in enumerate(agent_res.proposals[:slots_left]):
            counts["slots_used"] += 1
            try:
                res = evaluate_candidate(cur, raw, ctx, self.eval_edit, cand_budget,
                                         self.run_dir / f"it{it:02d}_c{j}", self.log)
                results.append((j, raw, res, None))
            except CandidateBudgetExhausted:
                break
            except Exception as err:  # noqa: BLE001 - one bad proposal never crashes the run
                results.append((j, raw, None, f"{type(err).__name__}: {err}"))
        eligible = [(r.metrics["val_edit_loss"], r.validation.n_scalars, j) for j, _, r, _ in results
                    if r is not None and r.status == "evaluated" and self._eligible(cur_eval, r)]
        winner = min(eligible)[2] if eligible else None
        chosen = None
        for j, raw, res, error in results:
            decision, d_loss, d_acc = "invalid", None, None
            if res is not None and res.status == "evaluated":
                d_loss = res.metrics["val_edit_loss"] - cur_eval.loss
                d_acc = res.metrics["val_edit_accuracy"] - cur_eval.accuracy
                decision = "accepted" if j == winner else "rejected"
                if j == winner:
                    chosen = (res.model, res.eval_result, res)
            elif res is not None and res.status == "no_edit":
                decision = "no_edit"
            history.append(EditHistoryItem(iteration=it, proposal_summary=summarise(raw), decision=decision,
                                           val_edit_loss_change=d_loss, val_edit_accuracy_change=d_acc))
            v = res.validation if res else None
            self._write({"type": "candidate", "iteration": it, "candidate_index": j, "proposal": raw,
                         "status": res.status if res else "error", "decision": decision,
                         "errors": (v.errors if v else []) + ([error] if error else []),
                         "n_scalars": v.n_scalars if v else 0, "rel_norm": v.rel_norm if v else 0.0,
                         "cumulative_after": v.cumulative_after if v else 0,
                         "edit_npz": res.edit_record.npz_path if res and res.edit_record else None,
                         "checkpoint_sha256": res.edit_record.state_sha256 if res and res.edit_record else None,
                         "val_edit": res.metrics if res and res.metrics else None, "val_edit_loss_change": d_loss,
                         "val_edit_accuracy_change": d_acc, "val_select": None})
        return chosen

    def _finish(self, states: list[StateRecord], best_index: int, best_state: dict[str, torch.Tensor],
                stop_reason: str, counts: dict[str, int], cand_budget: CandidateBudget, wall0: float) -> RunResult:
        final = clone_model(self.start)
        final.load_state_dict(best_state)
        diff = diff_stats(self.start, final)
        usages = getattr(self.agent, "all_usages", []) if self.agent else []
        cost = {"llm_calls": len(usages), "real_calls": sum(not u.cached for u in usages),
                "input_tokens": sum(u.input_tokens for u in usages), "output_tokens": sum(u.output_tokens for u in usages),
                "cost_usd": sum(u.cost_usd for u in usages), "wall_seconds": time.perf_counter() - wall0,
                "model": usages[0].model if usages else None,
                "prompt_version": usages[0].prompt_version if usages else None,
                "prompt_sha256": usages[0].prompt_sha256 if usages else None}
        counts = {**counts, "evaluations": cand_budget.evaluations_used,
                  "validator_rejections": cand_budget.validator_rejections, "no_edits": cand_budget.no_edits}
        meta = {"arm": self.s.arm, "seed": self.seed, "config_hash": config_hash(self.cfg),
                "final_state_index": best_index, "states": [asdict(r) for r in states], "counts": counts}
        saved = save_delta(self.ckpt_dir / DELTA_NAME, self.start, final, meta)
        sha = saved["final_state_sha256"]
        summary = {"arm": self.s.arm, "seed": self.seed, "tags": self.tags, "start_sha256": self.start_sha256,
                   "settings": asdict(self.s), "stop_reason": stop_reason, "final_index": best_index,
                   "states": [asdict(r) for r in states], "counts": counts, "cost": cost,
                   "checkpoint_sha256": sha, "delta_path": str(self.ckpt_dir / DELTA_NAME),
                   "delta_file_sha256": saved["file_sha256"], "diff_global_rel_norm": diff["global_rel_norm"],
                   "diff_n_changed": diff["n_changed"], "diff_per_layer": diff["per_layer"]}
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        self._write({"type": "end", "stop_reason": stop_reason, "final_index": best_index})
        return RunResult(self.s.arm, self.seed, stop_reason, states, best_index, final, diff, counts, cost, sha, self.run_dir)
