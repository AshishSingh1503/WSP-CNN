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
    manifest = make_splits(cfg)
    for name, info in manifest["files"].items():
        print(f"{name:11s} n={info['count']:6d} sha256={info['sha256']}")
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


EXPERIMENTS: dict[str, Callable[[Config, JsonlLogger, argparse.Namespace], None]] = {
    "make_splits": run_make_splits,
    "check_data": run_check_data,
    "baseline": run_baseline_cmd,
    "finetune": run_finetune_cmd,
}


def main() -> None:
    """Parse arguments and dispatch to the selected experiment."""
    parser = argparse.ArgumentParser(description="WSP-CNN")
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
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
