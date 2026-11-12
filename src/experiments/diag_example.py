"""Build example Pure/Hybrid diagnostic packets for a seed, twice each, and check determinism."""
from __future__ import annotations

import json
from typing import Any

import torch

from src.data.dataset import ensure_diag_indices, get_diag_batches, get_loader
from src.data.splits import load_indices
from src.diagnostics.packet import build_hybrid_packet, build_pure_packet, to_prompt_json
from src.models.checkpoint import require_common_checkpoint
from src.training.evaluate import evaluate
from src.utils.config import Config, resolve_path
from src.utils.logger import JsonlLogger
from src.utils.reproducibility import enable_deterministic, set_seed


def run_diag_example(cfg: Config, seed: int, log: JsonlLogger) -> dict[str, Any]:
    """Write results/examples/{pure,hybrid}_packet_seed{S}.json and report size and determinism."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.training.deterministic:
        enable_deterministic(cfg.training.warn_only_deterministic)
    set_seed(seed)
    ensure_diag_indices(cfg)
    model, _ = require_common_checkpoint(cfg, seed, device)
    result = evaluate(model, get_loader("val_edit", cfg), device, amp=False, num_classes=cfg.data.num_classes)
    kwargs = dict(eval_result=result, val_edit_indices=load_indices(resolve_path(cfg.paths.splits_dir), "val_edit"),
                  diag_batches=get_diag_batches(cfg), device=device, cfg=cfg, seed=seed, iteration=0,
                  remaining_edit_budget=int(cfg.model.expected_total_params * cfg.budgets.edit_budget_fraction),
                  remaining_candidate_budget=cfg.budgets.max_candidates, edit_history=[])
    out_dir = resolve_path(cfg.paths.examples_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    for name, build in (("pure", build_pure_packet), ("hybrid", build_hybrid_packet)):
        first, second = build(model, **kwargs), build(model, **kwargs)
        text = to_prompt_json(first, cfg.diagnostics.max_chars, cfg.diagnostics.sig_figs, log)
        same_text = text == to_prompt_json(second, cfg.diagnostics.max_chars, cfg.diagnostics.sig_figs)
        (out_dir / f"{name}_packet_seed{seed}.json").write_text(json.dumps(json.loads(text), indent=1), encoding="utf-8")
        report[name] = {"chars": len(text), "tokens_estimate": len(text) // 4, "truncated": "_truncated" in text,
                        "identical_prompt_json_across_two_runs": same_text,
                        "identical_raw_packet_across_two_runs": first.model_dump_json() == second.model_dump_json()}
        print(name, report[name])
    print("pure packet excerpt:", json.dumps(json.loads((out_dir / f'pure_packet_seed{seed}.json').read_text()), )[:900])
    log.log("diag_example", **report)
    return report
