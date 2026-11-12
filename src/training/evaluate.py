"""Model evaluation: accuracy, top-5, loss, macro P/R/F1, per-class accuracy, confusion matrix, predictions."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EvalResult:
    """Metrics for one pass over a loader. ``predictions`` are excluded from ``to_dict``."""

    accuracy: float
    top5_accuracy: float
    loss: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_accuracy: list[float]
    confusion_matrix: list[list[int]]
    n: int
    seconds: float
    images_per_sec: float
    predictions: np.ndarray = field(repr=False)
    labels: np.ndarray = field(repr=False)
    confidences: np.ndarray = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable metrics (no per-example arrays)."""
        return {k: v for k, v in self.__dict__.items() if k not in ("predictions", "labels", "confidences")}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str | torch.device,
             amp: bool = False, num_classes: int = 10) -> EvalResult:
    """Evaluate ``model`` in eval mode. ``amp`` defaults to fp32 (PROTOCOL.md Section 3)."""
    device = torch.device(device)
    model.eval()
    _sync(device)
    start = time.perf_counter()
    logits_all, labels_all = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
        logits_all.append(logits.float().cpu())
        labels_all.append(labels)
    _sync(device)
    seconds = time.perf_counter() - start
    return _summarise(torch.cat(logits_all), torch.cat(labels_all), num_classes, seconds)


def _summarise(logits: torch.Tensor, labels: torch.Tensor, num_classes: int, seconds: float) -> EvalResult:
    """Compute all metrics from logits and labels."""
    n = len(labels)
    preds = logits.argmax(1)
    top5 = logits.topk(min(5, num_classes), dim=1).indices.eq(labels[:, None]).any(1).float().mean().item()
    loss = torch.nn.functional.cross_entropy(logits, labels).item()
    y, p = labels.numpy(), preds.numpy()
    cm = np.bincount(y * num_classes + p, minlength=num_classes**2).reshape(num_classes, num_classes)
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, labels=range(num_classes),
                                                       average="macro", zero_division=0)
    per_class = (np.diag(cm) / np.maximum(cm.sum(1), 1)).tolist()
    return EvalResult(accuracy=float((y == p).mean()), top5_accuracy=top5, loss=loss,
                      macro_precision=float(prec), macro_recall=float(rec), macro_f1=float(f1),
                      per_class_accuracy=per_class, confusion_matrix=cm.tolist(), n=n,
                      seconds=seconds, images_per_sec=n / seconds if seconds > 0 else 0.0,
                      predictions=p.astype(np.int64), labels=y.astype(np.int64),
                      confidences=logits.softmax(1).max(1).values.numpy().astype(np.float64))


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 10) -> EvalResult:
    """Same metrics from precomputed logits (used for cached-feature evaluation)."""
    return _summarise(logits.float(), labels, num_classes, seconds=0.0)
