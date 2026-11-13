"""Protocol freeze: collect every final checkpoint hash, refuse unless clean and audited, write the lock."""
from __future__ import annotations

import json
import os
import stat
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.experiments.audit import run_audit
from src.experiments.run_all import PC_SUFFIX, is_done
from src.experiments.wsp_common import run_dirs
from src.models.checkpoint import read_registry
from src.utils.config import REPO_ROOT, Config, config_hash, load_faults, resolve_path
from src.utils.hashing import sha256_file

MAIN_ARMS = ("E2", "E3", "E5", "E6", "E6_RS", "E7")


def git_state(runner: Callable[..., Any] = subprocess.run) -> tuple[str, bool]:
    """(HEAD commit, working tree dirty?). Untracked-but-not-ignored files count as dirty."""
    head = runner(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    status = runner(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    return head, bool(status.strip())


def _delta_entry(arm: str, summary_path: Path, **extra: Any) -> dict[str, Any]:
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    return {"arm": arm, "seed": s["seed"], "kind": "delta", "path": s["delta_path"],
            "file_sha256": s["delta_file_sha256"], "state_sha256": s["checkpoint_sha256"],
            "start_sha256": s["start_sha256"], "start_kind": "faulted" if s.get("tags", {}).get("pc_fault") else "common", **extra}


def collect_entries(cfg: Config, seeds: list[int]) -> list[dict[str, Any]]:
    """Every checkpoint that will be evaluated on the test set, with the hashes that identify it."""
    entries: list[dict[str, Any]] = []
    registry = read_registry(cfg)
    faults = list(load_faults().faults)
    for seed in seeds:
        reg = registry[str(seed)]
        entries.append({"arm": "E0", "seed": seed, "kind": "full", "path": reg["path"], "file_sha256": reg["file_sha256"],
                        "state_sha256": reg["state_dict_sha256"], "start_sha256": None, "start_kind": None})
        e1 = json.loads((resolve_path(cfg.paths.e1_results_dir) / f"seed{seed}" / "metrics.json").read_text())
        best = resolve_path(cfg.paths.finetune_dir) / f"seed{seed}" / "best.pt"
        entries.append({"arm": "E1", "seed": seed, "kind": "full", "path": str(best), "file_sha256": sha256_file(best),
                        "state_sha256": e1["checkpoint_sha256"], "start_sha256": e1["start_checkpoint_sha256"], "start_kind": "common"})
        for arm in MAIN_ARMS:
            entries.append(_delta_entry(arm, run_dirs(cfg, arm, seed)[0] / "summary.json"))
        e4_path = run_dirs(cfg, "E4", seed)[0] / "summary.json"
        e4 = json.loads(e4_path.read_text(encoding="utf-8"))
        for k, sample in enumerate(e4["samples"]):
            entries.append({"arm": "E4", "seed": seed, "kind": "e4_rebuild", "sample": k, "path": str(e4_path),
                            "file_sha256": sha256_file(e4_path), "state_sha256": sample["checkpoint_sha256"],
                            "start_sha256": e4["start_sha256"], "start_kind": "common"})
        for fault in faults:
            arms = [f"PC_{fault}{PC_SUFFIX[m]}" for m in cfg.run_all.pc_methods]
            faulted_sha = None
            for arm in arms:
                entry = _delta_entry(arm, run_dirs(cfg, arm, seed)[0] / "summary.json", pc_fault=fault)
                entries.append(entry)
                faulted_sha = faulted_sha or entry["start_sha256"]
            entries.append({"arm": f"PC_{fault}_faulted", "seed": seed, "kind": "faulted_rebuild", "pc_fault": fault,
                            "state_sha256": faulted_sha, "start_sha256": reg["state_dict_sha256"], "start_kind": "common"})
    return sorted(entries, key=lambda e: (e["seed"], e["arm"], e.get("sample", -1)))


def build_lock(cfg: Config, seeds: list[int], commit: str, env: dict[str, Any]) -> dict[str, Any]:
    """The lock document (everything that must not change before the test set is opened)."""
    splits = resolve_path(cfg.paths.splits_dir)
    prompts = {Path(p).name: sha256_file(resolve_path(p)) for p in (cfg.llm.prompt_pure, cfg.llm.prompt_hybrid)}
    return {"created_at": datetime.now().isoformat(timespec="seconds"), "git_commit": commit, "config_hash": config_hash(cfg),
            "config": cfg.model_dump(mode="json"), "seeds": seeds, "prompts": prompts,
            "dsl_doc_sha256": sha256_file(REPO_ROOT / "docs" / "EDIT_DSL.md"),
            "split_manifest_sha256": sha256_file(splits / "manifest.json"),
            "diag_manifest_sha256": sha256_file(resolve_path(cfg.paths.diag_manifest)),
            "faults_yaml_sha256": sha256_file(resolve_path(cfg.paths.faults)),
            "protocol_md_sha256": sha256_file(REPO_ROOT / "PROTOCOL.md"),
            "registry": read_registry(cfg), "env": env, "checkpoints": collect_entries(cfg, seeds)}


def freeze(cfg: Config, seeds: list[int] | None = None, git: Callable[[], tuple[str, bool]] = git_state,
           audit: Callable[..., tuple[Any, list[str]]] = run_audit) -> str:
    """Write results/FROZEN_PROTOCOL.lock and return its SHA-256 (the final-eval token)."""
    seeds = seeds or list(cfg.seeds.primary)
    lock_path = resolve_path(cfg.paths.frozen_lock)
    if lock_path.exists():
        raise SystemExit(f"REFUSING: {lock_path} already exists; the protocol is frozen once")
    missing = [(s, st) for s in seeds for st in cfg.run_all.order if not is_done(cfg, st, s)]
    if missing:
        raise SystemExit(f"REFUSING: runs not complete (no valid done.json): {missing[:12]}{' ...' if len(missing) > 12 else ''}")
    _, failures = audit(cfg)
    if failures:
        raise SystemExit("REFUSING: budget audit failed:\n  - " + "\n  - ".join(failures))
    commit, dirty = git()
    if dirty:
        raise SystemExit("REFUSING: the git working tree is dirty; commit (or stash) everything first")
    env_path = resolve_path(cfg.paths.results_dir) / "env.json"
    if not env_path.is_file():
        raise SystemExit("REFUSING: results/env.json is missing; run python scripts/record_env.py and commit it")
    lock = build_lock(cfg, seeds, commit, json.loads(env_path.read_text(encoding="utf-8")))
    lock_path.write_text(json.dumps(lock, indent=1, sort_keys=True), encoding="utf-8")
    os.chmod(lock_path, stat.S_IREAD)
    return sha256_file(lock_path)
