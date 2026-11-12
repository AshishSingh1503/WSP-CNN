"""Stratified split properties: sizes, disjointness, coverage, balance, determinism, hash checks."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.data.splits import (SPLIT_NAMES, class_counts, ensure_splits, load_indices,
                             stratified_split, verify_splits, write_splits)
from src.utils.config import load_config
from src.utils.hashing import sha256_file

CFG = load_config().data
LABELS = np.repeat(np.arange(10), 5000).astype(np.int64)  # CIFAR-10 train has 5000 per class


@pytest.fixture(scope="module")
def splits() -> dict[str, np.ndarray]:
    return stratified_split(LABELS, CFG.n_val_edit, CFG.n_val_select, CFG.split_seed)


def test_sizes(splits) -> None:
    assert (len(splits["train"]), len(splits["val_edit"]), len(splits["val_select"])) == (45000, 2500, 2500)


def test_disjoint(splits) -> None:
    sets = [set(splits[n].tolist()) for n in SPLIT_NAMES]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


def test_union_is_full_train_set(splits) -> None:
    union = np.concatenate([splits[n] for n in SPLIT_NAMES])
    assert sorted(union.tolist()) == list(range(50000))


def test_val_splits_250_per_class(splits) -> None:
    for name in ("val_edit", "val_select"):
        assert class_counts(LABELS, splits[name], 10) == [250] * 10
    assert class_counts(LABELS, splits["train"], 10) == [4500] * 10


def test_rerun_identical_files(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    write_splits(a, LABELS, CFG)
    write_splits(b, LABELS, CFG)
    for name in SPLIT_NAMES:
        assert sha256_file(a / f"{name}_indices.npy") == sha256_file(b / f"{name}_indices.npy")
    before = sha256_file(a / "train_indices.npy")
    ensure_splits(a, LABELS, CFG)  # existing files are verified, not regenerated
    assert sha256_file(a / "train_indices.npy") == before
    assert json.loads((a / "manifest.json").read_text())["files"]["train"]["count"] == 45000


def test_different_seed_differs(splits) -> None:
    other = stratified_split(LABELS, CFG.n_val_edit, CFG.n_val_select, CFG.split_seed + 1)
    assert not np.array_equal(other["val_edit"], splits["val_edit"])


def test_tampered_file_detected(tmp_path: Path) -> None:
    write_splits(tmp_path, LABELS, CFG)
    np.save(tmp_path / "val_edit_indices.npy", load_indices(tmp_path, "val_edit")[::-1].copy())
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_splits(tmp_path, LABELS, CFG)


def test_partial_split_set_refused(tmp_path: Path) -> None:
    write_splits(tmp_path, LABELS, CFG)
    (tmp_path / "val_select_indices.npy").unlink()
    with pytest.raises(FileExistsError):
        ensure_splits(tmp_path, LABELS, CFG)


def test_unknown_split_name_rejected(tmp_path: Path) -> None:
    write_splits(tmp_path, LABELS, CFG)
    with pytest.raises(ValueError):
        load_indices(tmp_path, "test")
