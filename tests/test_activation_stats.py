"""Activation statistics on a hand-built two-conv network with known inputs."""
import numpy as np
import torch
from torch import nn

from src.diagnostics.activation_analyzer import (ActivationHooks, analyze_activations, class_selectivity,
                                                 summarize_activations)
from src.utils.config import load_config

CFG = load_config().diagnostics.model_copy(update={"selectivity_layers": ()})
IMG0 = [[1.0, -2.0], [3.0, 0.0]]
IMG1 = [[-1.0, 2.0], [0.0, 4.0]]


def _net() -> nn.Sequential:
    net = nn.Sequential(nn.Conv2d(1, 2, 1), nn.ReLU(), nn.Conv2d(2, 1, 1), nn.ReLU())
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([1.0, -1.0]).reshape(2, 1, 1, 1))
        net[0].bias.zero_()
        net[2].weight.copy_(torch.tensor([1.0, 1.0]).reshape(1, 2, 1, 1))
        net[2].bias.fill_(-1.0)
    return net


def _batches():  # two batches of one image: statistics must accumulate across batches
    return [(torch.tensor([[IMG0]]), torch.tensor([0])), (torch.tensor([[IMG1]]), torch.tensor([1]))]


def test_hand_computed_channel_stats() -> None:
    net = _net()
    rep = analyze_activations(net, _batches(), "cpu", CFG, num_classes=2)
    x = np.array(IMG0 + IMG1, dtype=float).reshape(-1)  # 1,-2,3,0,-1,2,0,4
    conv1 = rep.layers["0"]
    assert conv1["positions"] == 8
    assert np.allclose(conv1["mean"], [0.875, -0.875])
    assert np.allclose(conv1["inactive_fraction"], [0.5, 0.75])
    assert np.allclose(conv1["std"], [x.std(), (-x).std()])
    relu1 = rep.layers["1"]
    assert np.allclose(relu1["mean"], [1.25, 0.375])
    assert np.allclose(relu1["inactive_fraction"], [0.5, 0.75])
    assert np.allclose(relu1["mean_abs"], relu1["mean"])
    assert np.allclose(relu1["std"], [np.maximum(x, 0).std(), np.maximum(-x, 0).std()])
    conv2, relu2 = rep.layers["2"], rep.layers["3"]
    assert np.allclose(conv2["mean"], [0.625]) and np.allclose(conv2["inactive_fraction"], [0.5])
    assert np.allclose(relu2["mean"], [0.875]) and np.allclose(relu2["inactive_fraction"], [0.5])


def test_hooks_removed() -> None:
    net = _net()
    with ActivationHooks(net) as hooks:
        assert all(len(m._forward_hooks) == 1 for m in net if isinstance(m, (nn.Conv2d, nn.ReLU)))
        assert all(a.count == 0 for a in hooks.acc.values())  # nothing accumulated before a forward pass
    assert all(len(m._forward_hooks) == 0 for m in net.modules())
    analyze_activations(net, _batches(), "cpu", CFG, num_classes=2)
    assert all(len(m._forward_hooks) == 0 for m in net.modules())


def test_hooks_removed_after_exception() -> None:
    net = _net()
    try:
        with ActivationHooks(net):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert all(len(m._forward_hooks) == 0 for m in net.modules())


def test_dead_vs_sparse_separated() -> None:
    cfg = CFG.model_copy(update={"dead_threshold": 0.99, "sparse_threshold": 0.9})
    net = nn.Sequential(nn.Linear(2, 3), nn.ReLU())
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]))
        net[0].bias.copy_(torch.tensor([0.0, -100.0, 0.0]))  # channel 1 always inactive
    x = torch.zeros(100, 2)
    x[:, 0] = 1.0                      # channel 0 always active
    x[:5, 1] = 1.0                     # channel 2 active for 5% -> 95% inactive (sparse, not dead)
    rep = analyze_activations(net, [(x, torch.zeros(100, dtype=torch.long))], "cpu", cfg, num_classes=2)
    summary = summarize_activations(rep, cfg)["layers"]["1"]
    assert summary["dead_channels"] == [1] and summary["n_dead"] == 1
    assert summary["sparse_channels"] == [2] and summary["n_sparse"] == 1


def test_class_selectivity_finds_selective_channel() -> None:
    g = torch.Generator().manual_seed(0)
    pooled = torch.randn(60, 4, generator=g, dtype=torch.double) * 0.1
    labels = torch.arange(60) % 3
    pooled[labels == 1, 2] += 5.0  # channel 2 fires for class 1 only
    top = class_selectivity(pooled, labels, 3, top=2)
    assert top[1][0]["channel"] == 2 and top[1][0]["selectivity"] > 10
    assert top[0][0]["channel"] != 2 or top[0][0]["selectivity"] < 1
