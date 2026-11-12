"""Torch-free hashing helpers."""
from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()
