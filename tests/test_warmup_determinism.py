"""Head warm-up is deterministic for a fixed seed (CPU, tiny subset)."""
import torch

from src.training.warmup import train_head
from src.utils.config import E0Config, load_config

CPU = torch.device("cpu")


def _data(n: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"features": torch.randn(n, 4096, generator=g).half(),
            "labels": torch.randint(0, 10, (n,), generator=g)}


def _cfg() -> E0Config:
    return load_config().e0.model_copy(update={"epochs": 3, "batch_size": 32})


def test_same_seed_identical_head() -> None:
    tr, ve, vs = _data(200, 0), _data(60, 1), _data(60, 2)
    s1, h1 = train_head(tr, ve, vs, _cfg(), seed=11, device=CPU)
    s2, h2 = train_head(tr, ve, vs, _cfg(), seed=11, device=CPU)
    assert all(torch.equal(s1[k], s2[k]) for k in s1) and h1 == h2
    s3, _ = train_head(tr, ve, vs, _cfg(), seed=12, device=CPU)
    assert not torch.equal(s1["weight"], s3["weight"])


def test_history_and_selection() -> None:
    tr, ve, vs = _data(200, 0), _data(60, 1), _data(60, 2)
    _, hist = train_head(tr, ve, vs, _cfg(), seed=0, device=CPU)
    assert [h["epoch"] for h in hist] == [1, 2, 3]
    assert {"train_loss", "val_edit_acc", "val_select_acc", "lr"} <= hist[0].keys()
    assert hist[0]["lr"] == _cfg().lr and hist[1]["lr"] < hist[0]["lr"]
