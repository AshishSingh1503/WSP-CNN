"""E2: Pure WSP from the common checkpoint (no gradient information of any kind)."""
from __future__ import annotations

from src.experiments.arms import run_arm
from src.utils.config import Config
from src.utils.logger import JsonlLogger
from src.wsp.wsp_controller import RunResult


def run_pure_wsp(cfg: Config, seed: int, log: JsonlLogger, dry_run: bool = False) -> RunResult:
    """Run the Pure WSP loop for one seed and print the per-candidate table."""
    return run_arm(cfg, seed, log, "E2", "pure", dry_run=dry_run)
