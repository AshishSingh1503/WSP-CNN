"""The test set must be unreachable without the lock file, and unreferenced by non-final code."""
import ast
from pathlib import Path

import pytest
import yaml

from src.utils.config import DEFAULT_CONFIG_PATH, REPO_ROOT, Config
from src.utils.hashing import sha256_file

SRC = REPO_ROOT / "src"
# Only these files may mention the guarded loader / the raw test split.
ALLOWED = {SRC / "data" / "dataset.py", SRC / "experiments" / "final_eval.py"}
FORBIDDEN = "get_test_loader"


def violations(source: str) -> list[str]:
    """Static findings in one module's source: references to the test loader or train=False."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.alias) and FORBIDDEN in node.name:
            found.append(f"imports {node.name}")
        elif isinstance(node, ast.alias) and node.name == "*":
            found.append("star import")
        elif isinstance(node, ast.Name) and node.id == FORBIDDEN:
            found.append("name reference")
        elif isinstance(node, ast.Attribute) and node.attr == FORBIDDEN:
            found.append("attribute reference")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and FORBIDDEN in node.value:
            found.append("string reference")
        elif (isinstance(node, ast.keyword) and node.arg == "train"
              and isinstance(node.value, ast.Constant) and node.value.value is False):
            found.append("train=False (raw test-set construction)")
    return found


def test_scanner_detects_violations() -> None:
    assert violations("from src.data.dataset import get_test_loader")
    assert violations("import src.data.dataset as d\nd.get_test_loader(t, c)")
    assert violations("getattr(m, 'get_test_loader')")
    assert violations("from src.data.dataset import *")
    assert violations("CIFAR10(root='x', train=False)")
    assert not violations("from src.data.dataset import get_loader")


def test_no_module_outside_allowlist_references_test_loader() -> None:
    offenders = {}
    for path in SRC.rglob("*.py"):
        if path in ALLOWED:
            continue
        found = violations(path.read_text(encoding="utf-8"))
        if found:
            offenders[str(path.relative_to(REPO_ROOT))] = found
    assert not offenders, offenders


def test_wsp_and_experiments_dirs_exist_and_are_scanned() -> None:
    assert (SRC / "wsp").is_dir() and (SRC / "experiments").is_dir()
    assert SRC / "wsp" / "__init__.py" not in ALLOWED


# --- runtime guard (needs torch/torchvision) ---

def _cfg_with_lock(lock: Path) -> Config:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["paths"]["frozen_lock"] = str(lock)
    return Config.model_validate(raw)


def test_get_test_loader_raises_without_lock(tmp_path: Path) -> None:
    from src.data.dataset import get_test_loader
    with pytest.raises(PermissionError):
        get_test_loader("anything", _cfg_with_lock(tmp_path / "missing.lock"))


def test_get_test_loader_rejects_wrong_or_malformed_token(tmp_path: Path) -> None:
    from src.data.dataset import get_test_loader
    lock = tmp_path / "FROZEN_PROTOCOL.lock"
    lock.write_text("frozen")
    cfg = _cfg_with_lock(lock)
    for bad in ("", "0" * 64, None):
        with pytest.raises(PermissionError):
            get_test_loader(bad, cfg)  # type: ignore[arg-type]


def test_get_test_loader_accepts_correct_token(tmp_path: Path, monkeypatch) -> None:
    import torch
    from torch.utils.data import TensorDataset

    import src.data.dataset as ds
    lock = tmp_path / "FROZEN_PROTOCOL.lock"
    lock.write_text("frozen")
    monkeypatch.setattr(ds, "_build_test_dataset", lambda cfg: TensorDataset(torch.zeros(4, 1)))
    loader = ds.get_test_loader(sha256_file(lock), _cfg_with_lock(lock))
    assert len(loader.dataset) == 4


def test_get_loader_refuses_test_split() -> None:
    from src.data.dataset import get_loader
    from src.utils.config import load_config
    with pytest.raises(ValueError):
        get_loader("test", load_config())
