"""Macro F0.5 over Source 1 entities, including singletons."""

from __future__ import annotations

BETA = 0.5
BETA_SQ = BETA * BETA


def f_beta(precision: float, recall: float, beta_sq: float = BETA_SQ) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def score_entity(predicted: set[str], truth: set[str]) -> float:
    if not truth and not predicted:
        return 1.0
    if not truth:
        return 0.0
    if not predicted:
        return 0.0
    overlap = len(predicted & truth)
    precision = overlap / len(predicted)
    recall = overlap / len(truth)
    return f_beta(precision, recall)


def macro_prf(predictions: dict[str, set[str]], truth: dict[str, set[str]]) -> dict[str, float]:
    """Macro precision, recall, and F0.5. Empty-vs-empty counts as perfect."""
    keys = list(truth)
    if not keys:
        raise ValueError("ground truth is empty")
    precision_sum = 0.0
    recall_sum = 0.0
    for key in keys:
        predicted = predictions.get(key, set())
        actual = truth[key]
        if not predicted and not actual:
            precision_sum += 1.0
            recall_sum += 1.0
        elif not predicted:
            precision_sum += 1.0
        elif not actual:
            recall_sum += 0.0
        else:
            overlap = len(predicted & actual)
            precision_sum += overlap / len(predicted)
            recall_sum += overlap / len(actual)
    precision = precision_sum / len(keys)
    recall = recall_sum / len(keys)
    f_sum = sum(score_entity(predictions.get(key, set()), truth[key]) for key in keys)
    return {
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f05": f_sum / len(keys),
        "entities": float(len(keys)),
        "singletons": float(sum(1 for key in keys if not truth[key])),
    }


def macro_f05(predictions: dict[str, set[str]], truth: dict[str, set[str]]) -> dict[str, float]:
    keys = sorted(set(truth))
    if not keys:
        raise ValueError("ground truth is empty")
    scores = [score_entity(predictions.get(key, set()), truth[key]) for key in keys]
    return {
        "macro_f05": sum(scores) / len(scores),
        "entities": float(len(keys)),
        "singletons": float(sum(1 for key in keys if not truth[key])),
    }
