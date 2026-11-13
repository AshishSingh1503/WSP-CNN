"""Run the validation-only ablations (seeds 0-2): python scripts/run_ablations.py [--ids A1 A2 ...] [--yes] [--plan-only]."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.ablations import run_ablations  # noqa: E402
from src.experiments.run_all import prevent_sleep  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> int:
    """Parse arguments and run the planned ablations."""
    cfg = load_config()
    p = argparse.ArgumentParser(description="Ablations A1-A9 (exploratory, validation only)")
    p.add_argument("--ids", nargs="*", default=None, choices=list(cfg.ablations.defs))
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--yes", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()
    if not a.plan_only:
        prevent_sleep()
    return run_ablations(cfg, a.ids, a.seeds, a.yes, a.plan_only)


if __name__ == "__main__":
    sys.exit(main())
