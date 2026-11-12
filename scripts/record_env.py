"""Write results/env.json with Python, PyTorch, CUDA, GPU, driver and pip-freeze info."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.env import collect_env  # noqa: E402


def main() -> None:
    """Collect the environment and save it to ``results/env.json``."""
    out_path = REPO_ROOT / "results" / "env.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(collect_env(), indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
