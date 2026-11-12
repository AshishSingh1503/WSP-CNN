"""WSP-CNN command line entry point (make_splits, check_data, baseline)."""
from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path
from typing import Callable

from src.utils.config import DEFAULT_CONFIG_PATH, Config, config_hash, load_config, resolve_path
from src.utils.logger import JsonlLogger


def run_make_splits(cfg: Config, log: JsonlLogger, args: argparse.Namespace) -> None:
    """Create (or verify) the stratified split files and print the manifest summary."""
    from src.data.dataset import make_splits
    from src.data.dataset import ensure_diag_indices
    manifest = make_splits(cfg)
    for name, info in manifest["files"].items():
        print(f"{name:11s} n={info['count']:6d} sha256={info['sha256']}")
    print(f"diag subset: {len(ensure_diag_indices(cfg))} val_edit images")
    log.log("make_splits", manifest=manifest)


def run_check_data(cfg: Config, log: JsonlLogger, args: argparse.Namespace) -> None:
    """Verify splits, print counts/class balance, one batch shape and one val_edit epoch time."""
    from src.data.dataset import get_loader, make_splits
    manifest = make_splits(cfg)  # verifies hashes if files already exist
    for name, info in manifest["files"].items():
        print(f"{name:11s} n={info['count']:6d} per-class={info['class_counts']}")
    images, labels = next(iter(get_loader("train", cfg, seed=0)))
    print(f"train batch: images {tuple(images.shape)} {images.dtype}, labels {tuple(labels.shape)}")
    loader = get_loader("val_edit", cfg)
    start, n = time.perf_counter(), 0
    for images, _ in loader:
        n += len(images)
    elapsed = time.perf_counter() - start
    print(f"val_edit epoch: {n} images in {elapsed:.2f}s (num_workers={cfg.loader.num_workers})")
    log.log("check_data", val_edit_epoch_seconds=elapsed, val_edit_images=n)


def run_baseline_cmd(cfg: Config, log: JsonlLogger, args: argparse.Namespace) -> None:
    """E0: create or verify the common checkpoint for ``--seed``."""
    from src.experiments.baseline import run_baseline
    run_baseline(cfg, args.seed, log, reproduce_check=args.reproduce_check)


def run_finetune_cmd(cfg: Config, log: JsonlLogger, args: argparse.Namespace) -> None:
    """E1: full fine-tuning for ``--seed`` (or the seed-0 learning-rate sweep with ``--sweep``)."""
    from src.experiments.finetune import run_finetune, run_sweep
    if args.sweep:
        run_sweep(cfg, log)
    else:
        run_finetune(cfg, args.seed, log)


def run_diag_example_cmd(cfg: Config, log: JsonlLogger, args: argparse.Namespace) -> None:
    """Write example Pure/Hybrid packets for ``--seed`` and check determinism."""
    from src.experiments.diag_example import run_diag_example
    run_diag_example(cfg, args.seed, log)


def run_pure_wsp_cmd(cfg: Config, log: JsonlLogger, args: argparse.Namespace) -> None:
    """E2: Pure WSP for ``--seed`` (``--dry-run`` uses the scripted FakeLLM)."""
    from src.experiments.pure_wsp import run_pure_wsp
    run_pure_wsp(cfg, args.seed, log, dry_run=args.dry_run)


def run_positive_control_cmd(cfg: Config, log: JsonlLogger, args: argparse.Namespace) -> None:
    """Positive control for ``--seed``: all faults, or just ``--fault NAME``."""
    from src.experiments.positive_control import run_positive_control
    run_positive_control(cfg, args.seed, log, fault=args.fault, dry_run=args.dry_run, method=args.method)


def _arm_cmd(arm: str, method: str, layers: tuple[str, ...] | None = None):
    """Build an experiment function for one controller-driven arm."""
    def run(cfg: Config, log: JsonlLogger, args: argparse.Namespace) -> None:
        from src.experiments.arms import run_arm
        run_arm(cfg, args.seed, log, arm, method, layers, dry_run=args.dry_run)
    run.__doc__ = f"{arm}: {method} arm for --seed."
    return run


def run_random_edit_cmd(cfg: Config, log: JsonlLogger, args: argparse.Namespace) -> None:
    """E4: random edit sets matched to the E2 result of --seed."""
    from src.experiments.random_edit import run_random_edit
    run_random_edit(cfg, args.seed, log)


def run_audit_cmd(cfg: Config, log: JsonlLogger, args: argparse.Namespace) -> None:
    """Budget audit over all run folders; exits non-zero on failure."""
    from src.experiments.audit import main_audit
    main_audit(cfg)


EXPERIMENTS: dict[str, Callable[[Config, JsonlLogger, argparse.Namespace], None]] = {
    "make_splits": run_make_splits,
    "check_data": run_check_data,
    "baseline": run_baseline_cmd,
    "finetune": run_finetune_cmd,
    "diag_example": run_diag_example_cmd,
    "pure_wsp": run_pure_wsp_cmd,
    "positive_control": run_positive_control_cmd,
    "hybrid_wsp": _arm_cmd("E3", "hybrid"),
    "random_edit": run_random_edit_cmd,
    "random_search": _arm_cmd("E5", "random"),
    "last_layer": _arm_cmd("E6", "pure", ("classifier.6",)),
    "last_layer_rs": _arm_cmd("E6_RS", "random", ("classifier.6",)),
    "grad_heuristic": _arm_cmd("E7", "grad"),
    "audit": run_audit_cmd,
}


def main() -> None:
    """Parse arguments and dispatch to the selected experiment."""
    parser = argparse.ArgumentParser(description="WSP-CNN")
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true", help="WSP arms: use the scripted FakeLLM (no API calls)")
    parser.add_argument("--method", default="pure", choices=["pure", "hybrid", "random", "grad", "all"],
                        help="positive_control: which proposer to run on the faulted model ('all' = pure, random, grad)")
    parser.add_argument("--fault", default=None, help="positive_control: a single fault name from configs/faults.yaml")
    parser.add_argument("--sweep", action="store_true", help="finetune: run the seed-0 lr sweep")
    parser.add_argument("--reproduce-check", action="store_true",
                        help="baseline: retrain in memory and compare the hash with the registered one")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_id = f"{args.experiment}-{uuid.uuid4().hex[:8]}"
    log_path = resolve_path(cfg.paths.results_dir) / "logs" / f"{run_id}.jsonl"
    with JsonlLogger(log_path, run_id, args.seed, config_hash(cfg)) as log:
        EXPERIMENTS[args.experiment](cfg, log, args)


if __name__ == "__main__":
    main()
