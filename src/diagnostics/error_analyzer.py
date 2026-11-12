"""Error analysis from an EvalResult over val_edit (text-only; no images)."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.training.evaluate import EvalResult
from src.utils.config import DiagnosticsConfig


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int) -> float:
    """ECE with equal-width confidence bins (upper edge inclusive)."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def per_class_prf(cm: np.ndarray) -> list[dict[str, float]]:
    """Precision, recall, F1 for each class from a confusion matrix (rows = true)."""
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    return [{"class": i, "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
            for i in range(len(tp))]


def top_confused_pairs(cm: np.ndarray, k: int) -> list[dict[str, int]]:
    """The k largest off-diagonal cells (true -> predicted), ties broken by position."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(-off.reshape(-1), kind="stable")[:k]
    n = cm.shape[1]
    return [{"true": int(o // n), "predicted": int(o % n), "count": int(off.reshape(-1)[o])} for o in order]


def analyze_errors(result: EvalResult, indices: np.ndarray, cfg: DiagnosticsConfig) -> dict[str, Any]:
    """Confusion summary, calibration and a bounded list of the most confident mistakes.

    ``indices`` are the val_edit sample indices in evaluation order.
    """
    cm = np.asarray(result.confusion_matrix)
    correct = (result.predictions == result.labels).astype(float)
    wrong = np.flatnonzero(correct == 0)
    worst = wrong[np.lexsort((indices[wrong], -result.confidences[wrong]))][:cfg.max_misclassified]
    return {"confusion_matrix": cm.tolist(), "per_class": per_class_prf(cm),
            "top_confused_pairs": top_confused_pairs(cm, cfg.confusion_top_pairs),
            "mean_confidence_correct": float(result.confidences[correct == 1].mean()) if correct.any() else 0.0,
            "mean_confidence_wrong": float(result.confidences[correct == 0].mean()) if len(wrong) else 0.0,
            "ece": expected_calibration_error(result.confidences, correct, cfg.ece_bins),
            "misclassified": [{"index": int(indices[i]), "true": int(result.labels[i]),
                               "predicted": int(result.predictions[i]),
                               "confidence": float(result.confidences[i])} for i in worst]}
