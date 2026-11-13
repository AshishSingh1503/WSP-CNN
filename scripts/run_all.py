"""Run E0, E1, PC, E2-E7 for seeds 0-4 (resumable): python scripts/run_all.py [--yes] [--plan-only]."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.run_all import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
