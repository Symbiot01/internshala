"""Original 04_decision keep mask: per-entity rank cap, no source-cap / exempt."""

from __future__ import annotations

import numpy as np

import decide


def keep_mask_v04(packed: dict, threshold: float, cap: int, alpha: float, gate: float) -> np.ndarray:
    return (
        (packed["maxima"] >= gate)
        & (packed["scores"] >= threshold)
        & (packed["scores"] >= alpha * packed["maxima"])
        & (packed["ranks"] <= cap)
    )


def select_v04(packed: dict, threshold: float, cap: int, alpha: float, gate: float):
    keep = keep_mask_v04(packed, threshold, cap, alpha, gate)
    return list(
        zip(
            packed["s1_ids"][keep].tolist(),
            packed["cand_ids"][keep].tolist(),
            packed["scores"][keep].tolist(),
        )
    )


def assign_v04(packed: dict, rule: dict) -> dict[str, list[str]]:
    triples = select_v04(packed, rule["threshold"], rule["cap"], rule["alpha"], rule["gate"])
    return decide.unique_mapping(triples)
