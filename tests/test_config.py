"""Config loading, immutability and hashing."""
import pytest
import yaml
from pydantic import ValidationError

from src.utils.config import DEFAULT_CONFIG_PATH, Config, config_hash, load_config


def _raw() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def test_loads_protocol_values() -> None:
    cfg = load_config()
    assert (cfg.data.n_train, cfg.data.n_val_edit, cfg.data.n_val_select) == (45000, 2500, 2500)
    assert cfg.data.split_seed == 2026 and cfg.data.image_size == 224
    assert cfg.seeds.primary == (0, 1, 2, 3, 4)
    assert cfg.budgets.max_candidates == 40 and cfg.acceptance.min_loss_decrease == 0.002


def test_frozen() -> None:
    cfg = load_config()
    with pytest.raises(ValidationError):
        cfg.data.split_seed = 1  # type: ignore[misc]


def test_unknown_key_rejected() -> None:
    raw = _raw()
    raw["data"]["surprise"] = 1
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_inconsistent_sizes_rejected() -> None:
    raw = _raw()
    raw["data"]["n_train"] = 44999
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_hash_stable_and_sensitive() -> None:
    assert config_hash(load_config()) == config_hash(load_config())
    raw = _raw()
    raw["data"]["split_seed"] = 7
    assert config_hash(Config.model_validate(raw)) != config_hash(load_config())
