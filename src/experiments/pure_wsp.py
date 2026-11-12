"""E2: Pure WSP from the common checkpoint (no gradient information of any kind)."""
from __future__ import annotations

import shutil
from typing import Any

import torch

from src.experiments.wsp_common import WSPData, make_agent, make_settings, print_run_table, run_dirs
from src.models.checkpoint import require_common_checkpoint
from src.utils.config import Config
from src.utils.logger import JsonlLogger
from src.utils.reproducibility import enable_deterministic, set_seed
from src.wsp.wsp_controller import RunResult, WSPController


def run_pure_wsp(cfg: Config, seed: int, log: JsonlLogger, dry_run: bool = False) -> RunResult:
    """Run the Pure WSP loop for one seed and print the per-candidate table."""
    arm = "DRY_E2" if dry_run else "E2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.training.deterministic:
        enable_deterministic(cfg.training.warn_only_deterministic)
    set_seed(seed)
    start, start_hash = require_common_checkpoint(cfg, seed, device)
    res_dir, ckpt_dir = run_dirs(cfg, arm, seed)
    if dry_run:  # only scripted dry runs may be wiped and repeated
        shutil.rmtree(res_dir, ignore_errors=True)
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    data = WSPData(cfg, device)
    settings = make_settings(cfg, arm, "pure")
    agent = make_agent(cfg, "pure", settings, log, dry_run)
    result = WSPController(cfg, settings, agent, start, data.eval_edit, data.eval_select, data.diag_batches,
                           data.val_edit_indices, seed, res_dir, ckpt_dir, log, device).run()
    print_run_table(res_dir / "history.jsonl")
    _print_summary(result, start_hash)
    return result


def _print_summary(r: RunResult, start_hash: str) -> dict[str, Any]:
    s0, sf = r.states[0], r.states[r.final_index]
    print(f"{r.arm} seed {r.seed}: stop={r.stop_reason}; accepted {r.counts['accepted']}/{r.counts['iterations']} iterations; "
          f"final state {r.final_index}")
    print(f"  val_edit   acc {s0.val_edit['accuracy']:.4f} -> {sf.val_edit['accuracy']:.4f}   loss {s0.val_edit['loss']:.4f} -> {sf.val_edit['loss']:.4f}")
    print(f"  val_select acc {s0.val_select['accuracy']:.4f} -> {sf.val_select['accuracy']:.4f}   loss {s0.val_select['loss']:.4f} -> {sf.val_select['loss']:.4f}")
    print(f"  edit size {r.diff['n_changed']:,} scalars, global rel norm {r.diff['global_rel_norm']:.2e}; counts {r.counts}")
    c = r.cost
    print(f"  LLM: {c['llm_calls']} calls ({c['real_calls']} real), {c['input_tokens']:,} in / {c['output_tokens']:,} out tokens, "
          f"${c['cost_usd']:.3f}, model {c['model']}, prompt {c['prompt_version']} {str(c['prompt_sha256'])[:12]}")
    return {}
