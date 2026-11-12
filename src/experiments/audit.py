"""Budget audit: every run folder must share the start checkpoint and respect the candidate/edit budgets."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from src.experiments.wsp_common import edit_budget_scalars
from src.utils.config import Config, resolve_path

COLUMNS = ["arm", "seed", "group", "start_sha256", "candidate_evaluations", "slots_used", "changed_scalars",
           "rel_update_norm", "edit_budget", "llm_calls", "input_tokens", "output_tokens", "cost_usd", "wall_seconds",
           "val_select_accuracy"]
TABLE_ARMS = ["E2", "E3", "E4", "E5", "E6", "E6_RS", "E7"]


def _final_val_select(summary: dict[str, Any]) -> float | None:
    if "states" in summary:
        return summary["states"][summary["final_index"]]["val_select"]["accuracy"]
    return summary.get("aggregate", {}).get("accuracy", {}).get("mean")


def collect_rows(results_dir: Path) -> list[dict[str, Any]]:
    """One row per run folder (results/<arm>/seed<S>/summary.json); scripted DRY_ runs are ignored."""
    rows = []
    for path in sorted(results_dir.glob("*/seed*/summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        if str(s["arm"]).startswith("DRY_") or str(s["arm"]).endswith("_DRY"):
            continue
        cost = s.get("cost", {})
        tags = s.get("tags", {})
        rows.append({"arm": s["arm"], "seed": s["seed"],
                     "group": f"PC:{tags['pc_fault']}" if tags.get("pc_fault") else "main",
                     "start_sha256": s["start_sha256"],
                     "candidate_evaluations": s.get("counts", {}).get("evaluations", 0),
                     "slots_used": s.get("counts", {}).get("slots_used", 0), "changed_scalars": s.get("diff_n_changed", 0),
                     "rel_update_norm": s.get("diff_global_rel_norm", 0.0), "edit_budget": s.get("settings", {}).get("edit_budget"),
                     "llm_calls": cost.get("llm_calls", 0), "input_tokens": cost.get("input_tokens", 0),
                     "output_tokens": cost.get("output_tokens", 0), "cost_usd": cost.get("cost_usd", 0.0),
                     "wall_seconds": cost.get("wall_seconds", 0.0), "val_select_accuracy": _final_val_select(s),
                     "_summary": s})
    return rows


def audit_rows(cfg: Config, rows: list[dict[str, Any]], common_hashes: dict[int, str]) -> list[str]:
    """Return the list of failures (empty = pass)."""
    failures: list[str] = []
    default_budget, b = edit_budget_scalars(cfg), cfg.budgets
    pc_hash: dict[tuple[str, int], str] = {}
    for r in rows:
        tag = f"{r['arm']} seed {r['seed']}"
        if r["group"] == "main":
            expected = common_hashes.get(int(r["seed"]))
            if expected is None:
                failures.append(f"{tag}: no registered common checkpoint for this seed")
            elif r["start_sha256"] != expected:
                failures.append(f"{tag}: start hash {r['start_sha256'][:12]} differs from the common checkpoint {expected[:12]}")
        else:
            first = pc_hash.setdefault((r["group"], int(r["seed"])), r["start_sha256"])
            if first != r["start_sha256"]:
                failures.append(f"{tag}: faulted start hash differs from the other arms of {r['group']}")
        if r["candidate_evaluations"] > b.max_candidates or r["slots_used"] > b.max_candidates:
            failures.append(f"{tag}: used {r['candidate_evaluations']} evaluations / {r['slots_used']} slots, budget {b.max_candidates}")
        budget = r["edit_budget"]
        if budget is None or budget != default_budget:
            failures.append(f"{tag}: edit budget {budget} does not match the configured {default_budget}")
        if r["changed_scalars"] > (budget or default_budget):
            failures.append(f"{tag}: changed {r['changed_scalars']} scalars, budget {budget}")
        iterations = r["_summary"].get("counts", {}).get("iterations", 0)
        if iterations > b.max_iterations:
            failures.append(f"{tag}: {iterations} iterations, limit {b.max_iterations}")
    return failures


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def comparison_table(cfg: Config, rows: list[dict[str, Any]], seed: int, results_dir: Path) -> str:
    """Seed comparison on val_select: arm | acc | loss | macro F1 | changed scalars | rel norm | val evals | LLM cost."""
    lines = [f"{'arm':<8}{'val_sel acc':>12}{'loss':>8}{'macro F1':>9}{'changed':>9}{'rel norm':>10}{'val evals':>10}{'LLM $':>8}"]

    def add(name: str, acc: float, loss: float, f1: float, changed: Any, norm: Any, evals: Any, cost: float) -> None:
        lines.append(f"{name:<8}{acc:>12.4f}{loss:>8.4f}{f1:>9.4f}{str(changed):>9}{(f'{norm:.1e}' if isinstance(norm, float) else str(norm)):>10}"
                     f"{str(evals):>10}{cost:>8.2f}")
    e0 = results_dir / "E0" / f"seed{seed}" / "metrics.json"
    if e0.is_file():
        m = json.loads(e0.read_text())["val_select"]
        add("E0", m["accuracy"], m["loss"], m["macro_f1"], 0, 0.0, 0, 0.0)
    e1 = results_dir / "E1" / f"seed{seed}" / "metrics.json"
    if e1.is_file():
        m1 = json.loads(e1.read_text())
        v = m1["val_select"]
        add("E1 (ref)", v["accuracy"], v["loss"], v["macro_f1"], m1["param_diff"]["n_changed"], m1["param_diff"]["global_rel_norm"],
            m1["compute"]["val_evaluations"], 0.0)
    for arm in TABLE_ARMS:
        r = next((x for x in rows if x["arm"] == arm and x["seed"] == seed and x["group"] == "main"), None)
        if r is None:
            lines.append(f"{arm:<8}{'(not run)':>12}")
            continue
        s = r["_summary"]
        if arm == "E4":
            a = s["aggregate"]
            add("E4 mean", a["accuracy"]["mean"], a["loss"]["mean"], a["macro_f1"]["mean"], r["changed_scalars"],
                r["rel_update_norm"], s["val_select_evals"], 0.0)
            lines.append(f"{'':<8}{'+-' + format(a['accuracy']['sd'], '.4f'):>12}  (sd over {len(s['samples'])} random edit sets)")
        else:
            f = s["states"][s["final_index"]]["val_select"]
            add(arm, f["accuracy"], f["loss"], f.get("macro_f1", float("nan")), r["changed_scalars"], r["rel_update_norm"],
                r["candidate_evaluations"] + 1, r["cost_usd"])
    return "\n".join(lines)


def run_audit(cfg: Config, results_dir: Path | None = None, out_csv: Path | None = None,
              common_hashes: dict[int, str] | None = None, table_seed: int = 0) -> tuple[list[dict[str, Any]], list[str]]:
    """Audit all run folders; write the CSV; print the seed comparison table; return (rows, failures)."""
    results_dir = results_dir or resolve_path(cfg.paths.results_dir)
    if common_hashes is None:
        from src.models.checkpoint import read_registry
        common_hashes = {int(k): v["state_dict_sha256"] for k, v in read_registry(cfg).items()}
    rows = collect_rows(results_dir)
    failures = audit_rows(cfg, rows, common_hashes)
    write_csv(rows, out_csv or results_dir / "budget_audit.csv")
    print(f"audited {len(rows)} run folders")
    print(comparison_table(cfg, rows, table_seed, results_dir))
    if failures:
        print("\nAUDIT FAILED:")
        for f in failures:
            print("  -", f)
    else:
        print("\nAUDIT PASSED: all arms share the common start hash and respect the candidate and edit budgets")
    return rows, failures


def main_audit(cfg: Config) -> None:
    """CLI entry: exit status 1 on any audit failure."""
    _, failures = run_audit(cfg)
    if failures:
        sys.exit(1)
