"""JSONL run logger (one event per line) plus console logging."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, TextIO

_RESERVED = {"ts", "run_id", "seed", "config_hash", "event"}


def get_console_logger(name: str = "wsp") -> logging.Logger:
    """Console logger with a single stream handler (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class JsonlLogger:
    """Append-only JSONL logger; each line carries timestamp, run id, seed and config hash."""

    def __init__(self, path: str | Path, run_id: str, seed: int, config_hash: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id, self.seed, self.config_hash = run_id, seed, config_hash
        self._fh: TextIO = self.path.open("a", encoding="utf-8")
        self._console = get_console_logger()

    def log(self, event: str, **fields: Any) -> None:
        """Write one event line and echo it to the console."""
        clash = _RESERVED & fields.keys()
        if clash:
            raise ValueError(f"reserved log field(s): {sorted(clash)}")
        record = {"ts": time.time(), "run_id": self.run_id, "seed": self.seed,
                  "config_hash": self.config_hash, "event": event, **fields}
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()
        self._console.info("%s %s", event, fields)

    def close(self) -> None:
        """Close the underlying file."""
        self._fh.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
