"""The challenge metric: F0.5 per Source 1 entity, averaged over all entities.

An entity with no true matches scores 1.0 for an empty prediction and 0.0 otherwise.
"""

from __future__ import annotations

import numpy as np

import config


def entity_scores(true_positives: np.ndarray, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Vectorized per-entity F-beta from integer counts."""
    beta_sq = config.BETA ** 2
    scores = np.zeros(len(actual), dtype=np.float64)
    scores[(actual == 0) & (predicted == 0)] = 1.0
    both = (actual > 0) & (predicted > 0) & (true_positives > 0)
    precision = true_positives[both] / predicted[both]
    recall = true_positives[both] / actual[both]
    scores[both] = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    return scores


def summarize(true_positives: np.ndarray, predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Macro F0.5 plus macro precision and recall (empty-vs-empty counts as perfect)."""
    precision = np.where(predicted > 0, true_positives / np.maximum(predicted, 1), 1.0)
    recall = np.where(actual > 0, true_positives / np.maximum(actual, 1), np.where(predicted > 0, 0.0, 1.0))
    return {
        "macro_f05": float(entity_scores(true_positives, predicted, actual).mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "entities": int(len(actual)),
        "singletons": int((actual == 0).sum()),
        "singleton_accuracy": float(((predicted == 0) & (actual == 0)).sum() / max((actual == 0).sum(), 1)),
    }


def f_beta_from_sets(predicted: set[str], actual: set[str]) -> float:
    """Reference implementation used to test the vectorized version."""
    if not actual and not predicted:
        return 1.0
    if not actual or not predicted:
        return 0.0
    hits = len(predicted & actual)
    if hits == 0:
        return 0.0
    precision, recall = hits / len(predicted), hits / len(actual)
    beta_sq = config.BETA ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
