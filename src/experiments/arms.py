"""Run any controller-driven arm (E2, E3, E5, E6, E6-RS, E7, PC variants) through the SAME controller."""
from __future__ import annotations

import shutil
from typing import Any

import torch
from torch import nn

from src.experiments.wsp_common import WSPData, make_agent, make_settings, print_run_table, run_dirs
from src.models.checkpoint import require_common_checkpoint
from src.utils.config import Config
from src.utils.logger import JsonlLogger
from src.utils.reproducibility import enable_deterministic, set_seed
from src.wsp.grad_heuristic import GradHeuristicAgent
from src.wsp.random_proposals import RandomSearchAgent
from src.wsp.wsp_controller import RunResult, WSPController

# method -> (controller mode, uses an LLM, early stop allowed)
METHODS = {"pure": ("pure", True, True), "hybrid": ("hybrid", True, True),
           "random": ("pure", False, False), "grad": ("hybrid", False, False)}
LAST_LAYER = ("classifier.6",)


def run_controller(cfg: Config, arm: str, method: str, seed: int, start: nn.Module, data: WSPData,
                   device: torch.device, log: JsonlLogger, allowed_layers: tuple[str, ...] | None = None,
                   dry_run: bool = False, tags: dict[str, Any] | None = None) -> RunResult:
    """One run of ``method`` from ``start`` (the common checkpoint, or a faulted copy for the positive control)."""
    mode, uses_llm, early_stop = METHODS[method]
    name = f"DRY_{arm}" if (dry_run and uses_llm) else arm
    res_dir, ckpt_dir = run_dirs(cfg, name, seed)
    if dry_run and uses_llm:
        shutil.rmtree(res_dir, ignore_errors=True)
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    settings = make_settings(cfg, name, mode, allowed_layers, early_stop)
    if uses_llm:
        agent: Any = make_agent(cfg, mode, settings, log, dry_run)
    elif method == "random":
        agent = RandomSearchAgent(cfg, settings.allowed_layers, settings.allowed_ops, seed, name)
    else:
        agent = GradHeuristicAgent(cfg, settings.allowed_layers, settings.allowed_ops, data.diag_batches, device)
    return WSPController(cfg, settings, agent, start, data.eval_edit, data.eval_select, data.diag_batches,
                         data.val_edit_indices, seed, res_dir, ckpt_dir, log, device,
                         tags={"method": method, **(tags or {})}).run()


def print_summary(r: RunResult) -> None:
    """Short end-of-run report."""
    s0, sf = r.states[0], r.states[r.final_index]
    print(f"{r.arm} seed {r.seed}: stop={r.stop_reason}; accepted {r.counts['accepted']}/{r.counts['iterations']} iterations; final state {r.final_index}")
    print(f"  val_edit   acc {s0.val_edit['accuracy']:.4f} -> {sf.val_edit['accuracy']:.4f}   loss {s0.val_edit['loss']:.4f} -> {sf.val_edit['loss']:.4f}")
    print(f"  val_select acc {s0.val_select['accuracy']:.4f} -> {sf.val_select['accuracy']:.4f}   loss {s0.val_select['loss']:.4f} -> {sf.val_select['loss']:.4f}")
    print(f"  edit size {r.diff['n_changed']:,} scalars, global rel norm {r.diff['global_rel_norm']:.2e}; counts {r.counts}")
    c = r.cost
    if c["llm_calls"]:
        print(f"  LLM: {c['llm_calls']} calls ({c['real_calls']} real), {c['input_tokens']:,} in / {c['output_tokens']:,} out tokens, "
              f"${c['cost_usd']:.3f}, model {c['model']}, prompt {c['prompt_version']} {str(c['prompt_sha256'])[:12]}")


def run_arm(cfg: Config, seed: int, log: JsonlLogger, arm: str, method: str, allowed_layers: tuple[str, ...] | None = None,
            dry_run: bool = False) -> RunResult:
    """Load the common checkpoint for ``seed``, run one arm, and print its table."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.training.deterministic:
        enable_deterministic(cfg.training.warn_only_deterministic)
    set_seed(seed)
    start, _ = require_common_checkpoint(cfg, seed, device)
    data = WSPData(cfg, device)
    result = run_controller(cfg, arm, method, seed, start, data, device, log, allowed_layers, dry_run)
    print_run_table(run_dirs(cfg, result.arm, seed)[0] / "history.jsonl")
    print_summary(result)
    return result
