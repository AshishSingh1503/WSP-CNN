"""Run every arm for every seed, resumably (done.json markers with hashes), with a cost estimate and progress table."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.experiments.wsp_common import run_dirs
from src.utils.config import Config, REPO_ROOT, config_hash, load_faults, resolve_path
from src.utils.hashing import sha256_file

EXPERIMENT = {"E0": "baseline", "E1": "finetune", "PC": "positive_control", "E2": "pure_wsp", "E3": "hybrid_wsp",
              "E4": "random_edit", "E5": "random_search", "E6": "last_layer", "E6_RS": "last_layer_rs", "E7": "grad_heuristic"}
ARM_STEPS = ("E2", "E3", "E5", "E6", "E6_RS", "E7")  # controller arms: history.jsonl appears once a run has begun
PC_SUFFIX = {"pure": "", "hybrid": "_E3", "random": "_E5", "grad": "_E7"}


def step_dir(cfg: Config, step: str, seed: int) -> Path:
    """Folder that holds the done.json marker of (step, seed)."""
    if step in ("E0", "E1"):
        return resolve_path(cfg.paths.results_dir) / step / f"seed{seed}"
    if step == "PC":
        return resolve_path(cfg.paths.pc_results_dir) / f"seed{seed}"
    return run_dirs(cfg, step, seed)[0]


def _arm_evidence(cfg: Config, arm: str, seed: int) -> dict[str, Any] | None:
    summary = run_dirs(cfg, arm, seed)[0] / "summary.json"
    if not summary.is_file():
        return None
    s = json.loads(summary.read_text(encoding="utf-8"))
    if arm == "E4":
        return {"start_sha256": s["start_sha256"], "samples": [x["checkpoint_sha256"] for x in s["samples"]],
                "summary_file_sha256": sha256_file(summary)}
    delta = Path(s["delta_path"])
    if not delta.is_file():
        return None
    return {"start_sha256": s["start_sha256"], "checkpoint_sha256": s["checkpoint_sha256"],
            "delta_file_sha256": sha256_file(delta)}


def evidence(cfg: Config, step: str, seed: int) -> dict[str, Any] | None:
    """Hashes proving (step, seed) finished, or None if it did not. Recomputed from files each time."""
    if step == "E0":
        from src.models.checkpoint import read_registry
        entry = read_registry(cfg).get(str(seed))
        ok = (step_dir(cfg, "E0", seed) / "metrics.json").is_file()
        return {"checkpoint_sha256": entry["state_dict_sha256"], "file_sha256": entry["file_sha256"]} if entry and ok else None
    if step == "E1":
        metrics = step_dir(cfg, "E1", seed) / "metrics.json"
        best = resolve_path(cfg.paths.finetune_dir) / f"seed{seed}" / "best.pt"
        if not (metrics.is_file() and best.is_file()):
            return None
        return {"checkpoint_sha256": json.loads(metrics.read_text())["checkpoint_sha256"], "file_sha256": sha256_file(best)}
    if step == "PC":
        out: dict[str, Any] = {}
        for fault in load_faults().faults:
            for method in cfg.run_all.pc_methods:
                ev = _arm_evidence(cfg, f"PC_{fault}{PC_SUFFIX[method]}", seed)
                if ev is None:
                    return None
                out[f"PC_{fault}{PC_SUFFIX[method]}"] = ev
        return out
    return _arm_evidence(cfg, step, seed)


def is_done(cfg: Config, step: str, seed: int) -> bool:
    """True only if the marker exists AND the files still hash to what the marker recorded."""
    marker = step_dir(cfg, step, seed) / "done.json"
    if not marker.is_file():
        return False
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    current = evidence(cfg, step, seed)
    return current is not None and recorded.get("evidence") == current


def mark_done(cfg: Config, step: str, seed: int) -> dict[str, Any]:
    """Write done.json for a finished (step, seed); raises if the expected outputs are missing."""
    ev = evidence(cfg, step, seed)
    if ev is None:
        raise RuntimeError(f"{step} seed {seed}: outputs are incomplete; not marking done")
    payload = {"step": step, "seed": seed, "completed_at": datetime.now().isoformat(timespec="seconds"),
               "config_hash": config_hash(cfg), "evidence": ev}
    path = step_dir(cfg, step, seed) / "done.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_plan(cfg: Config, seeds: list[int], steps: list[str]) -> list[tuple[int, str, bool]]:
    """(seed, step, already_done) in the order seeds x steps."""
    return [(s, st, is_done(cfg, st, s)) for s in seeds for st in cfg.run_all.order if st in steps]


def _measured_minutes(history: dict[str, Any], step: str, default: float) -> float:
    runs = [r["minutes"] for r in history.get("runs", []) if r["step"] == step and r["status"] == "done"]
    return sum(runs) / len(runs) if runs else default


def estimate(cfg: Config, plan: list[tuple[int, str, bool]], history: dict[str, Any], pilot: dict[str, Any] | None) -> dict[str, Any]:
    """Remaining work: runs, LLM calls/tokens/USD (from the seed-0 pilot, else caps), and hours."""
    pending = [(s, st) for s, st, done in plan if not done]
    llm_runs = sum((len(load_faults().faults) if st == "PC" else 1) for _, st in pending if st in cfg.run_all.llm_steps)
    minutes = sum(_measured_minutes(history, st, cfg.run_all.default_minutes[st]) for _, st in pending)
    if pilot:
        calls, tokens_in, tokens_out, usd = (llm_runs * pilot[k] for k in ("llm_calls", "input_tokens", "output_tokens", "cost_usd"))
        basis = "average of the seed-0 pilot run"
    else:
        calls = tokens_in = tokens_out = None
        usd = llm_runs * cfg.llm.max_usd_per_run
        basis = "UPPER BOUND from the per-run USD cap (no pilot summary found)"
    return {"pending_runs": len(pending), "llm_runs": llm_runs, "llm_calls": calls, "input_tokens": tokens_in,
            "output_tokens": tokens_out, "usd": usd, "usd_basis": basis, "gpu_hours": minutes / 60.0}


def load_pilot(cfg: Config) -> dict[str, Any] | None:
    path = run_dirs(cfg, "E2", 0)[0] / "summary.json"
    if not path.is_file():
        return None
    c = json.loads(path.read_text(encoding="utf-8"))["cost"]
    return {k: c.get(k, 0) for k in ("llm_calls", "input_tokens", "output_tokens", "cost_usd")}


def _history_path(cfg: Config) -> Path:
    return resolve_path(cfg.run_all.progress_json)


def load_history(cfg: Config) -> dict[str, Any]:
    path = _history_path(cfg)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"runs": []}


def write_progress(cfg: Config, plan: list[tuple[int, str, bool]], history: dict[str, Any], failed: set[tuple[int, str]] = frozenset()) -> str:  # type: ignore[assignment]
    """results/progress.md: a seed x step status grid plus a table of recorded runs."""
    seeds = sorted({s for s, _, _ in plan})
    steps = [st for st in cfg.run_all.order if any(p[1] == st for p in plan)]
    grid = ["| step | " + " | ".join(f"seed {s}" for s in seeds) + " |", "|---|" + "---|" * len(seeds)]
    for st in steps:
        cells = []
        for s in seeds:
            done = next(d for ss, tt, d in plan if ss == s and tt == st)
            cells.append("failed" if (s, st) in failed else ("done" if done else "pending"))
        grid.append(f"| {st} | " + " | ".join(cells) + " |")
    rows = ["| seed | step | status | minutes | LLM cost USD | hash |", "|---|---|---|---|---|---|"]
    for r in history.get("runs", []):
        rows.append(f"| {r['seed']} | {r['step']} | {r['status']} | {r['minutes']:.1f} | {r.get('cost_usd', 0.0):.2f} | {str(r.get('hash', ''))[:12]} |")
    text = f"# Progress ({datetime.now().isoformat(timespec='seconds')})\n\n" + "\n".join(grid) + "\n\n## Runs\n\n" + "\n".join(rows) + "\n"
    path = resolve_path(cfg.run_all.progress_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def _cost_of(cfg: Config, step: str, seed: int) -> float:
    arms = [step] if step != "PC" else [f"PC_{f}{PC_SUFFIX[m]}" for f in load_faults().faults for m in cfg.run_all.pc_methods]
    total = 0.0
    for arm in arms:
        p = run_dirs(cfg, arm, seed)[0] / "summary.json"
        if p.is_file():
            total += json.loads(p.read_text(encoding="utf-8")).get("cost", {}).get("cost_usd", 0.0)
    return total


STALL_EXIT_CODE = 124


def run_with_watchdog(cmd: list[str], log: Path, stall_seconds: float, poll_seconds: float, append: bool = False) -> int:
    """Run ``cmd`` writing to ``log``; kill it if the log is silent for ``stall_seconds`` (returns 124)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a" if append else "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT)
        last_size, last_change = -1, time.monotonic()
        while proc.poll() is None:
            time.sleep(poll_seconds)
            size = log.stat().st_size
            if size != last_size:
                last_size, last_change = size, time.monotonic()
            elif time.monotonic() - last_change > stall_seconds:
                proc.kill()
                proc.wait()
                fh.write(f"\nWATCHDOG: no output for {stall_seconds:.0f}s; process killed\n")
                return STALL_EXIT_CODE
        return proc.returncode


def run_step(cfg: Config, step: str, seed: int) -> int:
    """Run one (step, seed) in a fresh process (isolates crashes and frees memory); returns its exit code.

    A step whose log goes silent (hung data-loader workers) is killed; resumable steps are retried.
    """
    cmd = [sys.executable, str(REPO_ROOT / "main.py"), "--experiment", EXPERIMENT[step], "--seed", str(seed)]
    if step == "PC":
        cmd += ["--method", "all"]
    log = resolve_path(cfg.run_all.logs_dir) / f"{step}_seed{seed}.log"
    r = cfg.run_all
    code = 1
    for attempt in range(1 + r.stall_retries):
        code = run_with_watchdog(cmd, log, r.stall_minutes * 60, r.poll_seconds, append=attempt > 0)
        if code == 0:
            break
        resumable_stall = code == STALL_EXIT_CODE and step in r.resumable_steps
        early_failure = step in ARM_STEPS and not (step_dir(cfg, step, seed) / "history.jsonl").exists()  # nothing written yet
        if not (resumable_stall or early_failure) or attempt == r.stall_retries:
            break
        print(f"  {step} seed {seed}: failed (exit {code}) before leaving partial output; retrying", flush=True)
    return code


def print_estimate(est: dict[str, Any]) -> None:
    print(f"Remaining: {est['pending_runs']} runs ({est['llm_runs']} LLM runs)")
    if est["llm_calls"] is not None:
        print(f"  LLM calls ~{est['llm_calls']:.0f}, tokens ~{est['input_tokens']:,.0f} in / {est['output_tokens']:,.0f} out")
    print(f"  USD ~{est['usd']:.2f}  [{est['usd_basis']}]")
    print(f"  GPU/laptop time ~{est['gpu_hours']:.1f} h")


def run_all(cfg: Config, seeds: list[int], steps: list[str], assume_yes: bool, plan_only: bool = False,
            runner=run_step, input_fn=input) -> int:
    """Execute the plan; skips completed runs; stops at the first failure. Returns a process exit code."""
    plan = build_plan(cfg, seeds, steps)
    history = load_history(cfg)
    est = estimate(cfg, plan, history, load_pilot(cfg))
    print_estimate(est)
    write_progress(cfg, plan, history)
    if plan_only or est["pending_runs"] == 0:
        return 0
    if not assume_yes and input_fn("Proceed? [y/N] ").strip().lower() != "y":
        print("aborted")
        return 2
    failed: set[tuple[int, str]] = set()
    for i, (seed, step, done) in enumerate(plan):
        if done:
            continue
        print(f"[{datetime.now():%H:%M:%S}] {step} seed {seed} ...", flush=True)
        start = time.perf_counter()
        code = runner(cfg, step, seed)
        minutes = (time.perf_counter() - start) / 60
        record = {"seed": seed, "step": step, "minutes": minutes, "status": "done" if code == 0 else "failed"}
        if code == 0:
            try:
                payload = mark_done(cfg, step, seed)
                record["hash"] = next(_flat(payload["evidence"]), "")
                record["cost_usd"] = _cost_of(cfg, step, seed)
                plan[i] = (seed, step, True)
            except RuntimeError as err:
                record["status"], code = f"incomplete: {err}", 1
        history["runs"] = [r for r in history["runs"] if not (r["seed"] == seed and r["step"] == step)] + [record]
        _history_path(cfg).parent.mkdir(parents=True, exist_ok=True)
        _history_path(cfg).write_text(json.dumps(history, indent=1), encoding="utf-8")
        if code != 0:
            failed.add((seed, step))
        write_progress(cfg, plan, history, failed)
        if code != 0:
            print(f"STOPPING: {step} seed {seed} failed (log in {cfg.run_all.logs_dir}); fix and re-run to resume")
            return 1
    print("all planned runs complete")
    return 0


def _flat(ev: Any):
    if isinstance(ev, dict):
        for v in ev.values():
            yield from _flat(v)
    elif isinstance(ev, list):
        for v in ev:
            yield from _flat(v)
    elif isinstance(ev, str):
        yield ev


def prevent_sleep() -> bool:
    """Ask Windows not to sleep while this process runs (a per-process request; no system setting is changed)."""
    if sys.platform != "win32":
        return False
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))


def main(argv: list[str] | None = None) -> int:
    """CLI: python scripts/run_all.py [--seeds 0 1 ...] [--steps E0 ...] [--yes] [--plan-only]."""
    import argparse
    cfg = load_config_default()
    p = argparse.ArgumentParser(description="Run all arms for all seeds (resumable)")
    p.add_argument("--seeds", type=int, nargs="*", default=list(cfg.seeds.primary))
    p.add_argument("--steps", nargs="*", default=list(cfg.run_all.order), choices=list(cfg.run_all.order))
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.add_argument("--plan-only", action="store_true", help="print the estimate and progress table, run nothing")
    a = p.parse_args(argv)
    if not a.plan_only:
        print("sleep prevention requested:", prevent_sleep(), flush=True)
    return run_all(cfg, a.seeds, a.steps, a.yes, a.plan_only)


def load_config_default() -> Config:
    from src.utils.config import load_config
    return load_config()
