"""Shared fixtures. A session guard proves the real common checkpoint file is never modified by tests."""
import copy

import pytest
import torch

from src.models.alexnet import build_model
from src.models.checkpoint import read_registry
from src.utils.config import load_config, resolve_path
from src.utils.hashing import sha256_file


@pytest.fixture(scope="session", autouse=True)
def common_checkpoint_untouched():
    """G4: the registered seed-0 common checkpoint file must have the same SHA-256 after the whole session."""
    cfg = load_config()
    entry = read_registry(cfg).get("0")
    path = resolve_path(entry["path"]) if entry else None
    before = sha256_file(path) if path and path.is_file() else None
    yield
    if before is not None:
        assert sha256_file(path) == before, "common checkpoint file was modified during the test session"
        assert before == entry["file_sha256"], "common checkpoint file differs from its registry hash"


@pytest.fixture(scope="session")
def alexnet_template() -> torch.nn.Module:
    """Random-init AlexNet with a 10-class head (never edited directly; tests deep-copy it)."""
    torch.manual_seed(0)
    return build_model(pretrained=False, seed=0).eval()


@pytest.fixture()
def model(alexnet_template):
    """A fresh, editable copy of the template AlexNet."""
    return copy.deepcopy(alexnet_template)
