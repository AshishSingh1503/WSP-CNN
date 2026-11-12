"""Frozen, validated configuration for WSP-CNN, loaded from YAML."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


class _Frozen(BaseModel):
    """Base for all config sections: immutable, unknown keys rejected."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class PathsConfig(_Frozen):
    data_dir: str
    splits_dir: str
    checkpoints_dir: str
    results_dir: str
    prompts_dir: str
    frozen_lock: str
    features_dir: str
    baseline_dir: str
    registry: str
    model_card: str
    e0_results_dir: str


class DataConfig(_Frozen):
    num_classes: int
    n_train: int
    n_val_edit: int
    n_val_select: int
    n_official_train: int
    n_test: int
    split_seed: int
    native_image_size: int
    image_size: int
    crop_padding: int
    imagenet_mean: tuple[float, float, float]
    imagenet_std: tuple[float, float, float]

    @model_validator(mode="after")
    def _sizes_add_up(self) -> "DataConfig":
        total = self.n_train + self.n_val_edit + self.n_val_select
        if total != self.n_official_train:
            raise ValueError(f"split sizes sum to {total}, expected {self.n_official_train}")
        if self.n_val_edit % self.num_classes or self.n_val_select % self.num_classes:
            raise ValueError("val split sizes must be divisible by num_classes")
        return self


class LoaderConfig(_Frozen):
    batch_size_train: int
    batch_size_eval: int
    num_workers: int
    pin_memory: bool


class TrainingConfig(_Frozen):
    amp: bool
    eval_amp: bool
    deterministic: bool
    warn_only_deterministic: bool


class ModelConfig(_Frozen):
    arch: str
    pretrained_weights: str
    expected_total_params: int


class E0Config(_Frozen):
    optimizer: str
    epochs: int
    lr: float
    weight_decay: float
    batch_size: int
    schedule: str
    trainable_prefixes: tuple[str, ...]
    select_on: str
    feature_dtype: str
    agreement_images: int
    min_agreement: float


class E1Config(_Frozen):
    epochs: int
    lr: float
    momentum: float
    weight_decay: float
    schedule: str
    select_on: str


class BudgetConfig(_Frozen):
    max_iterations: int
    candidates_per_iteration: int
    max_candidates: int
    max_accepted_per_iteration: int
    edit_budget_fraction: float
    ablation_fractions: tuple[float, ...]
    max_llm_retries: int

    @model_validator(mode="after")
    def _slots_consistent(self) -> "BudgetConfig":
        if self.max_candidates != self.max_iterations * self.candidates_per_iteration:
            raise ValueError("max_candidates must equal max_iterations * candidates_per_iteration")
        return self


class AcceptanceConfig(_Frozen):
    min_loss_decrease: float
    require_accuracy_non_decrease: bool
    stop_after_unproductive: int
    disable_early_stop_for: tuple[str, ...]


class AddDeltaConfig(_Frozen):
    max_entries: int
    max_abs_sigma_frac: float


class ScaleChannelConfig(_Frozen):
    max_channels: int
    factor_min: float
    factor_max: float


class BiasShiftConfig(_Frozen):
    max_entries: int
    max_abs_classifier: float
    max_abs_other: float


class LowRankConfig(_Frozen):
    max_rows: int
    max_cols: int
    max_rank: int
    factor_abs_max: float
    max_abs_sigma_frac: float


class DslConfig(_Frozen):
    max_ops_per_candidate: int
    add_delta: AddDeltaConfig
    scale_output_channel: ScaleChannelConfig
    bias_shift: BiasShiftConfig
    low_rank_update: LowRankConfig


class ControlsConfig(_Frozen):
    e4_trials_per_seed: int
    e4_edits_per_trial: int
    e5_runs_per_seed: int
    e7_relative_steps: tuple[float, ...]
    e7_max_entries: int


class LLMConfig(_Frozen):
    provider: str
    model: str
    temperature: float
    max_tokens: int


class PositiveControlConfig(_Frozen):
    f1_tensor: str
    f1_channels: int
    f1_seed_offset: int
    f2_tensor: str
    f2_class_index: int
    f2_shift: float
    f3_tensor: str
    f3_channels: int
    f3_seed_offset: int
    min_val_select_drop_pp: float
    repair_success_fraction: float
    h4_null_rate: float


class SeedsConfig(_Frozen):
    primary: tuple[int, ...]
    pilot: tuple[int, ...]

    @model_validator(mode="after")
    def _disjoint(self) -> "SeedsConfig":
        if set(self.primary) & set(self.pilot):
            raise ValueError("pilot seeds must not overlap primary seeds")
        return self


class StatsConfig(_Frozen):
    n_bootstrap: int
    bootstrap_seed: int
    alpha: float
    n_primary_claims: int
    min_effect_pp: float
    min_positive_seeds: int


class Config(_Frozen):
    """Top-level frozen configuration."""

    paths: PathsConfig
    data: DataConfig
    loader: LoaderConfig
    training: TrainingConfig
    model: ModelConfig
    e0: E0Config
    e1: E1Config
    budgets: BudgetConfig
    acceptance: AcceptanceConfig
    dsl: DslConfig
    controls: ControlsConfig
    llm: LLMConfig
    positive_control: PositiveControlConfig
    seeds: SeedsConfig
    stats: StatsConfig


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate a YAML config file into a frozen ``Config``."""
    with Path(path).open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    return Config.model_validate(raw)


def config_hash(cfg: Config) -> str:
    """SHA-256 of the canonical JSON form of the config (key-order independent)."""
    canonical = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_path(rel: str | Path) -> Path:
    """Resolve a config path against the repo root (absolute paths pass through)."""
    return (REPO_ROOT / rel).resolve()
