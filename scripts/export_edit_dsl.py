"""Write docs/EDIT_DSL.md from the Pydantic schema and config."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.wsp.dsl_doc import render_dsl_doc  # noqa: E402


def main() -> None:
    """Render and save the DSL document."""
    out = REPO_ROOT / "docs" / "EDIT_DSL.md"
    out.write_text(render_dsl_doc(load_config()), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
