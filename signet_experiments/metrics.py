"""Threshold selection and verification metrics (label 1 means forged)."""

from __future__ import annotations

import numpy as np


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray]:
    labels = np.asarray(labels).astype(int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if labels.size == 0 or labels.size != probabilities.size:
        raise ValueError("Labels and probabilities must be non-empty and equal in size.")
    if not 0 <= threshold <= 1:
        raise ValueError("Classification threshold must be between 0 and 1.")

    predictions = probabilities >= threshold
    forged = labels == 1
    genuine = labels == 0
    tp = int(np.count_nonzero(forged & predictions))
    fn = int(np.count_nonzero(forged & ~predictions))
    fp = int(np.count_nonzero(genuine & predictions))
    tn = int(np.count_nonzero(genuine & ~predictions))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "accuracy": (tp + tn) / labels.size,
        "precision_forged": precision,
        "recall_forged": recall,
        "f1_forged": f1,
        "far": fn / (tp + fn) if tp + fn else 0.0,
        "frr": fp / (tn + fp) if tn + fp else 0.0,
    }
    return metrics, np.array([[tn, fp], [fn, tp]], dtype=int)


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    steps: int = 99,
) -> tuple[float, dict[str, float]]:
    if steps < 2:
        raise ValueError("At least two threshold candidates are required.")
    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    for threshold in np.linspace(0.01, 0.99, steps):
        metrics, _ = classification_metrics(labels, probabilities, float(threshold))
        if best_metrics is None or metrics["f1_forged"] > best_metrics["f1_forged"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_metrics is None:
        raise RuntimeError("Threshold selection did not evaluate any candidates.")
    return best_threshold, best_metrics
