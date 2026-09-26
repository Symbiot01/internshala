"""Per-entity decision rule and Unique Mapping Clustering.

The rule (tuned on the calibration half against true macro F0.5):
  * singleton gate: if the best probability is below ``s``, predict nothing
  * threshold: keep candidates with probability >= ``t``
  * relative margin: keep candidates with probability >= ``alpha`` * best
  * cap: keep at most ``k`` remaining candidates **per source**
  * cap exempt: score >= CAP_EXEMPT bypasses the cap (near-certain gold at rank 7+)
  * hard source maxima: at most SOURCE_CAP_S2 / SOURCE_CAP_S3 unless exempt

Unique Mapping Clustering (Papadakis, Efthymiou, Thanos, Hassanzadeh, EDBT 2022):
sort surviving (entity, candidate, score) triples by score and give each
Source 2/3 id to at most one Source 1 entity. This only removes matches.
"""

from __future__ import annotations

from itertools import product

import numpy as np

import config
import metrics


def pack(
    s1_ids: np.ndarray,
    cand_ids: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray | None = None,
    truth: dict[str, list[str]] | None = None,
) -> dict:
    """Sort by entity, then by score descending, and precompute ranks and maxima.

    ``truth`` is the official per-entity match list. Grid search must use that
    count, not the number of retrieved positives — otherwise blocking misses are
    scored as singletons (F=1) instead of misses (F=0).
    """
    source = _source_codes(cand_ids)
    order = np.lexsort((-scores, source, s1_ids))
    s1_ids, cand_ids, scores, source = s1_ids[order], cand_ids[order], scores[order], source[order]
    labels = labels[order] if labels is not None else None
    entities, starts, counts = np.unique(s1_ids, return_index=True, return_counts=True)
    entity_index = np.repeat(np.arange(len(entities)), counts)
    ranks = np.arange(len(s1_ids), dtype=np.int32) - np.repeat(starts, counts) + 1
    maxima = np.repeat(scores[starts], counts)
    source_change = np.empty(len(s1_ids), dtype=bool)
    source_change[0] = True
    if len(s1_ids) > 1:
        source_change[1:] = (s1_ids[1:] != s1_ids[:-1]) | (source[1:] != source[:-1])
    source_starts = np.flatnonzero(source_change)
    source_counts = np.diff(np.append(source_starts, len(s1_ids)))
    ranks_source = np.arange(len(s1_ids), dtype=np.int32) - np.repeat(source_starts, source_counts) + 1
    gold = np.zeros(len(entities), dtype=np.int32)
    if truth is not None:
        gold = np.array([len(truth.get(entity, [])) for entity in entities], dtype=np.int32)
    elif labels is not None:
        np.add.at(gold, entity_index, labels)
    return {
        "s1_ids": s1_ids,
        "cand_ids": cand_ids,
        "scores": scores,
        "source": source,
        "labels": labels,
        "entities": entities,
        "entity_index": entity_index,
        "ranks": ranks,
        "ranks_source": ranks_source,
        "maxima": maxima,
        "gold": gold,
    }


def _source_codes(cand_ids: np.ndarray) -> np.ndarray:
    codes = np.empty(len(cand_ids), dtype=np.int8)
    for index, cand_id in enumerate(cand_ids):
        codes[index] = 2 if str(cand_id).startswith("S2-") else 3
    return codes


def keep_mask(
    packed: dict,
    threshold: float,
    cap: int,
    alpha: float,
    gate: float,
    cap_exempt: float | None = None,
    source_cap_s2: int | None = None,
    source_cap_s3: int | None = None,
) -> np.ndarray:
    if cap_exempt is None:
        cap_exempt = config.CAP_EXEMPT
    if source_cap_s2 is None:
        source_cap_s2 = config.SOURCE_CAP_S2
    if source_cap_s3 is None:
        source_cap_s3 = config.SOURCE_CAP_S3
    hard = np.where(packed["source"] == 2, min(cap, source_cap_s2), min(cap, source_cap_s3))
    cap_ok = packed["ranks_source"] <= hard
    if cap_exempt > 0:
        cap_ok = cap_ok | (packed["scores"] >= cap_exempt)
    return (
        (packed["maxima"] >= gate)
        & (packed["scores"] >= threshold)
        & (packed["scores"] >= alpha * packed["maxima"])
        & cap_ok
    )


def rule_summary(packed: dict, threshold: float, cap: int, alpha: float, gate: float) -> dict[str, float]:
    keep = keep_mask(packed, threshold, cap, alpha, gate)
    predicted = np.zeros(len(packed["entities"]), dtype=np.int32)
    hits = np.zeros(len(packed["entities"]), dtype=np.int32)
    np.add.at(predicted, packed["entity_index"][keep], 1)
    if packed["labels"] is not None:
        np.add.at(hits, packed["entity_index"][keep & (packed["labels"] == 1)], 1)
    return metrics.summarize(hits, predicted, packed["gold"])


def grid_search(packed: dict) -> dict:
    """Search the four-parameter rule.

    For a well-calibrated binary classifier the F-beta-optimal threshold is
    approximately t* = F*/(1+β²) (Lipton, Elkan, Narayanaswamy, 2014). With
    β=0.5 that is 0.8·F*, so an F0.5 near 0.97 anchors t around 0.78. The grid
    still decides; the formula only sets the search centre. Hand & Christen
    (2018) discuss F-measure for record linkage more generally.
    """
    thresholds = (0.50, 0.55, 0.60, 0.65, 0.70, 0.74, 0.76, 0.80, 0.85, 0.90)
    caps = (1, 2, 3, 4, 5, 6, 8, 10, 12)
    alphas = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
    gates = (0.00, 0.30, 0.40, 0.50, 0.60, 0.70)
    best: dict | None = None
    for threshold, cap, alpha, gate in product(thresholds, caps, alphas, gates):
        summary = rule_summary(packed, threshold, cap, alpha, gate)
        row = {"threshold": threshold, "cap": cap, "alpha": alpha, "gate": gate, **summary}
        if best is None or row["macro_f05"] > best["macro_f05"]:
            best = row
    return best or {}


def select(packed: dict, threshold: float, cap: int, alpha: float, gate: float) -> list[tuple[str, str, float]]:
    keep = keep_mask(packed, threshold, cap, alpha, gate)
    return list(zip(
        packed["s1_ids"][keep].tolist(),
        packed["cand_ids"][keep].tolist(),
        packed["scores"][keep].tolist(),
    ))


def unique_mapping(triples: list[tuple[str, str, float]]) -> dict[str, list[str]]:
    """Greedy one-to-one assignment: highest score claims a candidate id first."""
    chosen: dict[str, list[str]] = {}
    claimed: set[str] = set()
    for s1_id, cand_id, _score in sorted(triples, key=lambda row: row[2], reverse=True):
        if cand_id in claimed:
            continue
        claimed.add(cand_id)
        chosen.setdefault(s1_id, []).append(cand_id)
    return chosen


def evaluate_assignment(
    assignment: dict[str, list[str]],
    entities: np.ndarray,
    truth: dict[str, list[str]],
) -> dict[str, float]:
    """Score every entity in ``entities``, including those with an empty assignment."""
    predicted = np.zeros(len(entities), dtype=np.int32)
    hits = np.zeros(len(entities), dtype=np.int32)
    actual = np.zeros(len(entities), dtype=np.int32)
    for index, entity in enumerate(entities):
        gold = set(truth.get(entity, []))
        pred = set(assignment.get(entity, []))
        predicted[index] = len(pred)
        hits[index] = len(pred & gold)
        actual[index] = len(gold)
    return metrics.summarize(hits, predicted, actual)


def breakdown(
    assignment: dict[str, list[str]],
    entities: np.ndarray,
    countries: np.ndarray,
    truth: dict[str, list[str]],
    cand_script: dict[str, bool] | None = None,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for country in sorted(set(countries.tolist())):
        member = countries == country
        out[f"country={country}"] = evaluate_assignment(assignment, entities[member], truth)
    if cand_script is not None:
        indic = np.array([
            any(cand_script.get(mid, False) for mid in truth.get(entity, []))
            for entity in entities
        ])
        out["script=indic_gold"] = evaluate_assignment(assignment, entities[indic], truth)
        out["script=latin_gold"] = evaluate_assignment(assignment, entities[~indic], truth)
    return out
