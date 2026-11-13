"""Freeze the protocol: python scripts/freeze_protocol.py  (prints the final_eval token)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.freeze import freeze  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    """Run the freeze checks and write results/FROZEN_PROTOCOL.lock."""
    token = freeze(load_config())
    print("FROZEN. final_eval_token (SHA-256 of results/FROZEN_PROTOCOL.lock):")
    print(token)


if __name__ == "__main__":
    main()
