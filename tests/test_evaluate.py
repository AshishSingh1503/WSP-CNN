"""evaluate(): metrics computed correctly on a model whose logits are its inputs."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.evaluate import evaluate


def test_metrics_known_values() -> None:
    labels = torch.tensor([0, 1, 2, 2, 1, 0])
    logits = torch.zeros(6, 3)
    for i, p in enumerate([0, 1, 2, 1, 1, 2]):  # samples 3 and 5 are wrong
        logits[i, p] = 5.0
    loader = DataLoader(TensorDataset(logits, labels), batch_size=4)
    r = evaluate(nn.Identity(), loader, "cpu", num_classes=3)
    assert r.n == 6 and np.isclose(r.accuracy, 4 / 6) and r.top5_accuracy == 1.0
    assert r.confusion_matrix == [[1, 0, 1], [0, 2, 0], [0, 1, 1]]
    assert np.allclose(r.per_class_accuracy, [0.5, 1.0, 0.5])
    assert r.predictions.tolist() == [0, 1, 2, 1, 1, 2] and r.labels.tolist() == labels.tolist()
    assert np.isclose(r.loss, nn.functional.cross_entropy(logits, labels).item())
    assert "predictions" not in r.to_dict()
