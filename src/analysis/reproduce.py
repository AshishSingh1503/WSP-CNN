"""Regenerate every table and figure from saved results, with a hash manifest to prove it is identical."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.analysis.figures import make_all
from src.analysis.stats import FilePredictions, Predictions
from src.analysis.tables import write_tables
from src.utils.config import Config, resolve_path
from src.utils.hashing import sha256_file

MANIFEST_NAME = "manifest.json"


def manifest_of(tables_dir: Path, figures_dir: Path) -> dict[str, str]:
    """SHA-256 of every generated table and figure file (keys are paths relative to results/)."""
    out: dict[str, str] = {}
    for root in (tables_dir, figures_dir):
        for path in sorted(root.glob("*")):
            if path.is_file() and path.name != MANIFEST_NAME:
                out[f"{root.name}/{path.name}"] = sha256_file(path)
    return out


def reproduce(cfg: Config, preds: Predictions, tables_dir: Path | None = None, figures_dir: Path | None = None,
              seeds: list[int] | None = None) -> dict[str, str]:
    """Write all tables and figures; returns (and stores next to the tables) the hash manifest."""
    tables_dir = tables_dir or resolve_path(cfg.analysis.tables_dir)
    figures_dir = figures_dir or resolve_path(cfg.analysis.figures_dir)
    res = write_tables(cfg, preds, tables_dir, seeds)
    make_all(cfg, preds, res, res["frames"], figures_dir)
    from src.experiments.ablations import ablation_table
    ablation_table(cfg).to_csv(tables_dir / "ablations.csv", index=False, float_format="%.10g")
    manifest = manifest_of(tables_dir, figures_dir)
    (tables_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return manifest


def check(cfg: Config, preds: Predictions, scratch: Path, seeds: list[int] | None = None) -> list[str]:
    """Regenerate into ``scratch`` and compare with the stored manifest; returns the list of differing files."""
    stored = json.loads((resolve_path(cfg.analysis.tables_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))
    fresh = reproduce(cfg, preds, scratch / "tables", scratch / "figures", seeds)
    return sorted(k for k in set(stored) | set(fresh) if stored.get(k) != fresh.get(k))


def default_predictions(cfg: Config) -> FilePredictions:
    """Predictions from results/test; raises a clear error if the final evaluation has not been run."""
    if not (resolve_path(cfg.paths.results_dir) / "test" / "eval_ledger.jsonl").is_file():
        raise SystemExit("ERROR: results/test/ is missing: run the final evaluation (main.py --experiment final_eval) first")
    return FilePredictions(cfg)
