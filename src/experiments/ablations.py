"""Stage 9 Part B ablations: each is a config override of the Pure WSP arm (validation only, exploratory)."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable

from src.experiments.arms import run_arm
from src.experiments.run_all import run_with_watchdog
from src.experiments.wsp_common import run_dirs
from src.utils.config import REPO_ROOT, Config, apply_overrides, resolve_path
from src.utils.logger import JsonlLogger
from src.wsp.wsp_controller import RunResult


def arm_name(cfg: Config, ablation_id: str) -> str:
    """Output arm name, e.g. ABL_A1 (results/ABL_A1/seed{S}/)."""
    return f"{cfg.ablations.arm_prefix}_{ablation_id}"


def ablation_config(cfg: Config, ablation_id: str) -> Config:
    """The base config with the ablation's overrides applied (validated)."""
    return apply_overrides(cfg, dict(cfg.ablations.defs[ablation_id].overrides))


def run_ablation(cfg: Config, seed: int, ablation_id: str, log: JsonlLogger, dry_run: bool = False) -> RunResult:
    """One ablation run for one seed (a normal controller run with different settings and its own output folder)."""
    d = cfg.ablations.defs[ablation_id]
    for key, value in d.env.items():
        os.environ[key] = value                       # e.g. LLM_MODEL for the second-LLM ablation
    layers = tuple(d.allowed_layers) if d.allowed_layers else None
    return run_arm(ablation_config(cfg, ablation_id), seed, log, arm_name(cfg, ablation_id), d.method, layers, dry_run)


def is_complete(cfg: Config, ablation_id: str, seed: int) -> bool:
    return (run_dirs(cfg, arm_name(cfg, ablation_id), seed)[0] / "summary.json").is_file()


def plan(cfg: Config, ids: list[str], seeds: list[int]) -> list[tuple[str, int, bool]]:
    """(ablation id, seed, already complete) for every planned run."""
    return [(i, s, is_complete(cfg, i, s)) for i in ids for s in seeds]


def estimate(cfg: Config, todo: int) -> dict[str, Any]:
    """Rough USD for ``todo`` runs from the seed-0 E2 pilot (or the per-run cap if there is no pilot)."""
    costs = []
    for seed in cfg.seeds.primary:
        path = run_dirs(cfg, "E2", seed)[0] / "summary.json"
        if path.is_file():
            costs.append(json.loads(path.read_text(encoding="utf-8")).get("cost", {}).get("cost_usd", 0.0))
    if costs:
        return {"runs": todo, "usd": todo * sum(costs) / len(costs), "basis": f"mean cost of {len(costs)} finished E2 runs"}
    return {"runs": todo, "usd": todo * cfg.llm.max_usd_per_run, "basis": "per-run cap (no E2 runs found)"}


def _subprocess_runner(cfg: Config) -> Callable[[str, int], int]:
    logs = resolve_path(cfg.run_all.logs_dir)

    def run(ablation_id: str, seed: int) -> int:
        cmd = [sys.executable, str(REPO_ROOT / "main.py"), "--experiment", "ablation", "--ablation", ablation_id, "--seed", str(seed)]
        return run_with_watchdog(cmd, logs / f"ABL_{ablation_id}_seed{seed}.log", cfg.run_all.stall_minutes * 60, cfg.run_all.poll_seconds)
    return run


def run_ablations(cfg: Config, ids: list[str] | None, seeds: list[int] | None, assume_yes: bool, plan_only: bool = False,
                  runner: Callable[[str, int], int] | None = None, input_fn: Callable[[str], str] = input) -> int:
    """Run every (ablation, seed) without a summary yet; asks for confirmation; stops at the first failure."""
    ids = ids or list(cfg.ablations.defs)
    seeds = seeds or list(cfg.ablations.seeds)
    items = plan(cfg, ids, seeds)
    todo = [(i, s) for i, s, done in items if not done]
    est = estimate(cfg, len(todo))
    print(f"{len(items)} ablation runs planned, {len(todo)} to run; about USD {est['usd']:.2f} [{est['basis']}]")
    if plan_only or not todo:
        return 0
    if not assume_yes and input_fn("Proceed? [y/N] ").strip().lower() != "y":
        print("aborted")
        return 2
    run = runner or _subprocess_runner(cfg)
    for ablation_id, seed in todo:
        print(f"{ablation_id} seed {seed} ...", flush=True)
        code = run(ablation_id, seed)
        if code != 0 or not is_complete(cfg, ablation_id, seed):
            print(f"STOPPING: {ablation_id} seed {seed} failed (exit {code}); re-run to resume")
            return 1
    return 0


def ablation_table(cfg: Config) -> Any:
    """Validation-only table: val_select accuracy, change vs E2, accepted edits and LLM cost per ablation (mean over seeds)."""
    import pandas as pd
    root, rows = resolve_path(cfg.paths.results_dir), []

    def load(arm: str, seed: int) -> dict[str, Any] | None:
        path = root / arm / f"seed{seed}" / "summary.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    entries = [(k, v[0], "reference") for k, v in cfg.ablations.references.items()] + [("A4_hybrid_E3", "E3", "reference")]
    entries += [(i, arm_name(cfg, i), "ablation") for i in cfg.ablations.defs]
    for label, arm, kind in entries:
        got = [(r, e2) for r, e2 in ((load(arm, s), load("E2", s)) for s in cfg.ablations.seeds) if r is not None and e2 is not None]
        if not got:
            continue
        acc = [r["states"][r["final_index"]]["val_select"]["accuracy"] for r, _ in got]
        base = [e["states"][e["final_index"]]["val_select"]["accuracy"] for _, e in got]
        n = len(got)
        rows.append({"ablation": label, "kind": kind, "arm": arm, "seeds": n, "val_select_acc_%": 100 * sum(acc) / n,
                     "vs_E2_pp": 100 * (sum(acc) - sum(base)) / n, "mean_accepted": sum(r["counts"]["accepted"] for r, _ in got) / n,
                     "mean_changed_scalars": sum(r["diff_n_changed"] for r, _ in got) / n,
                     "llm_usd_per_seed": sum(r["cost"].get("cost_usd", 0.0) for r, _ in got) / n})
    cols = ["ablation", "kind", "arm", "seeds", "val_select_acc_%", "vs_E2_pp", "mean_accepted", "mean_changed_scalars", "llm_usd_per_seed"]
    return pd.DataFrame(rows, columns=cols)
