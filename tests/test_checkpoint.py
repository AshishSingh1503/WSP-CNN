"""Checkpoint round trip, tamper detection, immutability, clone independence, registry."""
import os
import stat
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from src.models.checkpoint import (clone_model, load_checkpoint, read_registry, register_checkpoint,
                                   require_common_checkpoint, save_checkpoint, sidecar_path)
from src.utils.config import DEFAULT_CONFIG_PATH, Config
from src.utils.reproducibility import sha256_tensor_dict


def tiny(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))


def test_round_trip_identical(tmp_path: Path) -> None:
    model, path = tiny(), tmp_path / "m.pt"
    h = save_checkpoint(model, path, {"seed": 0})
    loaded = load_checkpoint(path, h, model=tiny(1))
    for (ka, a), (kb, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert ka == kb and torch.equal(a, b)
    assert sha256_tensor_dict(loaded.state_dict()) == h


def test_sidecar_and_read_only(tmp_path: Path) -> None:
    path = tmp_path / "m.pt"
    h = save_checkpoint(tiny(), path, {"seed": 7, "config_hash": "abc"})
    side = sidecar_path(path).read_text()
    assert h in side and '"seed": 7' in side and "env" in side
    assert not os.access(path, os.W_OK)
    with pytest.raises(FileExistsError):
        save_checkpoint(tiny(), path, {})


def test_tampered_file_fails(tmp_path: Path) -> None:
    path = tmp_path / "m.pt"
    h = save_checkpoint(tiny(), path, {})
    os.chmod(path, stat.S_IWRITE)
    torch.save(tiny(5).state_dict(), path)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(path, h, model=tiny())


def test_clone_independent() -> None:
    src = tiny()
    clone = clone_model(src)
    before = src[0].weight.clone()
    with torch.no_grad():
        clone[0].weight.add_(1.0)
    assert torch.equal(src[0].weight, before) and not torch.equal(clone[0].weight, before)


def _cfg(tmp_path: Path) -> Config:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["paths"]["registry"] = str(tmp_path / "registry.json")
    return Config.model_validate(raw)


def test_require_missing_seed_exits(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no common checkpoint registered"):
        require_common_checkpoint(_cfg(tmp_path), 0)


def test_registry_hash_checks(tmp_path: Path) -> None:
    cfg, path = _cfg(tmp_path), tmp_path / "m.pt"
    h = save_checkpoint(tiny(), path, {})
    register_checkpoint(cfg, 0, path, h)
    assert read_registry(cfg)["0"]["state_dict_sha256"] == h
    with pytest.raises(ValueError):  # a different hash for the same seed is refused
        register_checkpoint(cfg, 0, path, "1" * 64)
    os.chmod(path, stat.S_IWRITE)
    torch.save(tiny(9).state_dict(), path)  # tamper
    with pytest.raises(SystemExit, match="hash mismatch"):
        require_common_checkpoint(cfg, 0)
