"""Tiny CPU smoke tests for the fine-tuning loop, resume, and parameter-diff statistics."""
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.checkpoint import clone_model, save_checkpoint
from src.training.fine_tune import FitSettings, fit, lr_factor
from src.utils.logger import JsonlLogger
from src.utils.param_diff import diff_stats

CPU = torch.device("cpu")


class Tiny(nn.Module):
    """Has the 'features.*' / 'classifier.*' naming the optimizer groups rely on."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.features = nn.Linear(8, 6)
        self.classifier = nn.Linear(6, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.relu(self.features(x)))


def _data(n: int, seed: int) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    return TensorDataset(torch.randn(n, 8, generator=g), torch.randint(0, 10, (n,), generator=g))


def _loaders():
    train = lambda epoch: DataLoader(_data(64, 1), batch_size=16, shuffle=True,  # noqa: E731
                                     generator=torch.Generator().manual_seed(epoch))
    vals = {"val_select": DataLoader(_data(40, 2), batch_size=20), "val_edit": DataLoader(_data(40, 3), batch_size=20)}
    return train, vals


def _settings(epochs: int) -> FitSettings:
    return FitSettings(backbone_lr=1e-2, classifier_lr=1e-2, weight_decay=0.0, epochs=epochs,
                       warmup_epochs=1, accum_steps=1, amp=False, seed=0)


def _log(tmp: Path) -> JsonlLogger:
    return JsonlLogger(tmp / "log.jsonl", "test", 0, "hash")


def test_smoke_two_batches_one_epoch(tmp_path: Path) -> None:
    model, train, vals = Tiny(), *_loaders()
    reference = clone_model(model)
    with _log(tmp_path) as log:
        res = fit(model, train, vals, _settings(1), CPU, tmp_path / "run", log, max_batches=2)
    assert res.completed and len(res.history) == 1 and res.best_epoch == 1
    assert {"val_select_acc", "val_edit_acc", "train_loss", "gpu_seconds"} <= res.history[0].keys()
    assert res.compute["images_seen"] == 32 and res.compute["val_evaluations"] == 2
    best = clone_model(model)
    best.load_state_dict(res.best_state)
    h = save_checkpoint(best, tmp_path / "best.pt", {"seed": 0})
    assert (tmp_path / "best.pt").is_file() and len(h) == 64
    stats = diff_stats(reference, best)
    assert stats["n_changed"] > 0 and stats["global_rel_norm"] > 0
    assert set(stats["per_layer"]) == {"features", "classifier"}


def test_resume_matches_uninterrupted(tmp_path: Path) -> None:
    train, vals = _loaders()
    full, split = Tiny(), Tiny()
    with _log(tmp_path) as log:
        fit(full, train, vals, _settings(3), CPU, tmp_path / "a", log)
        fit(split, train, vals, _settings(3), CPU, tmp_path / "b", log, stop_after_epoch=1)
        resumed = fit(Tiny(), train, vals, _settings(3), CPU, tmp_path / "b", log)
    assert resumed.completed and [h["epoch"] for h in resumed.history] == [1, 2, 3]
    a = fit(Tiny(), train, vals, _settings(3), CPU, tmp_path / "a", _log(tmp_path))  # already finished
    assert a.completed
    assert [h["train_loss"] for h in a.history] == [h["train_loss"] for h in resumed.history]
    assert all(torch.equal(a.best_state[k], resumed.best_state[k]) for k in a.best_state)


def test_diff_stats_identical_is_zero() -> None:
    m = Tiny()
    s = diff_stats(m, clone_model(m))
    assert s["n_changed"] == 0 and s["global_rel_norm"] == 0.0
    assert s["n_total"] == sum(p.numel() for p in m.parameters())
    assert all(v["rel_norm"] == 0.0 for v in s["per_layer"].values())


def test_diff_stats_known_change() -> None:
    a, b = Tiny(), Tiny()
    with torch.no_grad():
        b.classifier.bias.add_(1.0)  # 10 scalars change
    s = diff_stats(a, b)
    assert s["n_changed"] == 10 and s["per_tensor"]["classifier.bias"]["n_changed"] == 10
    expected = (10 ** 0.5) / float(sum((p.double() ** 2).sum() for p in a.parameters()) ** 0.5)
    assert s["global_rel_norm"] == pytest.approx(expected, rel=1e-5)
    assert s["per_layer"]["features"]["n_changed"] == 0


def test_lr_schedule_shape() -> None:
    f = [lr_factor(i, 10, 100) for i in range(100)]
    assert f[0] == pytest.approx(0.1) and f[9] == pytest.approx(1.0) and f[10] == pytest.approx(1.0)
    assert all(f[i] >= f[i + 1] for i in range(10, 99)) and f[-1] < 1e-3
