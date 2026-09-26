"""Reciprocal Rank Fusion (Cormack, Clarke, Buettcher 2009) and recall reporting.

RRF score of a document = sum over lists of 1 / (k + rank), rank starting at 1.
It uses only ranks, so TF-IDF cosines and embedding cosines need no calibration.
"""

from __future__ import annotations

import numpy as np

import config


def rrf_fuse(ranked_lists: list[np.ndarray], top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Fuse per-query ranked position arrays (-1 = empty slot) into one top-k list."""
    positions = np.concatenate(ranked_lists, axis=1).astype(np.int64)
    scores = np.concatenate(
        [np.broadcast_to(1.0 / (config.RRF_K + np.arange(1, lst.shape[1] + 1)), lst.shape) for lst in ranked_lists],
        axis=1,
    ).astype(np.float64)
    scores = np.where(positions < 0, -np.inf, scores)

    # A document found by both lists appears twice: add its scores, then drop the copy.
    order = np.argsort(positions, axis=1, kind="stable")
    positions = np.take_along_axis(positions, order, axis=1)
    scores = np.take_along_axis(scores, order, axis=1)
    duplicate = (positions[:, 1:] == positions[:, :-1]) & (positions[:, 1:] >= 0)
    scores[:, :-1] += np.where(duplicate, scores[:, 1:], 0.0)
    scores[:, 1:][duplicate] = -np.inf

    take = min(top_k, positions.shape[1])
    best = np.argsort(-scores, axis=1, kind="stable")[:, :take]
    fused_positions = np.take_along_axis(positions, best, axis=1)
    fused_scores = np.take_along_axis(scores, best, axis=1)
    fused_positions[~np.isfinite(fused_scores)] = -1
    return fused_positions.astype(np.int32), np.where(np.isfinite(fused_scores), fused_scores, 0.0).astype(np.float32)


def recall_curve(positions: np.ndarray, gold: list[np.ndarray], ks: list[int]) -> dict[int, float]:
    """Share of gold pairs found in the first k positions (pairs counted over all queries)."""
    total = sum(len(g) for g in gold)
    found = {k: 0 for k in ks}
    for row, gold_positions in enumerate(gold):
        if len(gold_positions) == 0:
            continue
        ranks = np.flatnonzero(np.isin(positions[row], gold_positions))
        for k in ks:
            found[k] += int((ranks < k).sum())
    return {k: found[k] / max(total, 1) for k in ks}
