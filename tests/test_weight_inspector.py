"""Weight statistics agree with numpy."""
import numpy as np
import torch
from torch import nn

from src.diagnostics.weight_inspector import inspect_weights, tensor_stats, unit_stats


def test_tensor_stats_match_numpy() -> None:
    rng = np.random.default_rng(0)
    arr = rng.normal(0.3, 2.0, size=(7, 5, 3, 3))
    arr[0, 0, 0, 0] = 0.0
    arr[1, 1, 1, 1] = 40.0  # outlier beyond 3 std
    s = tensor_stats(torch.tensor(arr))
    std = arr.std()
    assert s["shape"] == [7, 5, 3, 3] and s["numel"] == arr.size
    assert np.isclose(s["mean"], arr.mean()) and np.isclose(s["std"], std)
    assert s["min"] == arr.min() and s["max"] == arr.max()
    assert np.isclose(s["l1"], np.abs(arr).sum()) and np.isclose(s["l2"], np.linalg.norm(arr))
    assert np.isclose(s["zero_fraction"], (arr == 0).mean())
    assert np.isclose(s["frac_abs_gt_3std"], (np.abs(arr) > 3 * std).mean())
    assert s["frac_abs_gt_3std"] > 0


def test_unit_stats_and_topk_bottomk() -> None:
    w = torch.zeros(5, 2, 2)
    for i in range(5):
        w[i] = (i + 1) * torch.tensor([[1.0, -1.0], [1.0, -1.0]])  # L2 = 2*(i+1)
    st = unit_stats(w)
    assert torch.allclose(st["l2"], torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0], dtype=torch.double))
    assert torch.allclose(st["mean"], torch.zeros(5, dtype=torch.double))
    model = nn.Sequential(nn.Conv2d(2, 5, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(w.reshape(5, 1, 2, 2).repeat(1, 2, 1, 1) / 2 ** 0.5)
    report = inspect_weights(model, topk=2)
    units = report["units"]["0"]
    assert units["n_units"] == 5
    assert [u["index"] for u in units["top"]] == [4, 3] and [u["index"] for u in units["bottom"]] == [0, 1]


def test_all_parameters_covered() -> None:
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    report = inspect_weights(model, topk=5)
    assert set(report["tensors"]) == {"0.weight", "0.bias", "2.weight", "2.bias"}
    assert len(report["units"]["0"]["top"]) == 3  # k capped at the number of units
