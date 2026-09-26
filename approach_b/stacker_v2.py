"""LightGBM stacker with reverse-competition features. Isolated from stacker.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config
import stacker as base

FEATURE_NAMES = base.FEATURE_NAMES + (
    "reverse_rank",
    "reverse_gap",
    "reverse_claimants",
)

STACKER_DIR = config.MODELS_DIR / "stacker_v2"


def reverse_features(s1_ids: np.ndarray, cand_ids: np.ndarray, s1: object, lookups: dict) -> dict[str, np.ndarray]:
    """Rank/gap/claimants of this Source 1 inside the candidate's reverse top-k."""
    import pandas as pd

    n = len(cand_ids)
    ranks = np.full(n, 99, dtype=np.float32)
    gaps = np.zeros(n, dtype=np.float32)
    claimants_n = np.zeros(n, dtype=np.float32)
    s1_row = pd.Series(s1_ids).map({i: p for p, i in enumerate(s1.id.to_numpy())}).to_numpy(dtype=np.float64)
    source = np.where(pd.Series(cand_ids).astype(str).str.startswith("S2-"), 2, 3)
    for src in (2, 3):
        pos_mat, score_mat, id_map = lookups[src]
        mask = source == src
        if not mask.any():
            continue
        rows = pd.Series(cand_ids[mask]).map(id_map).to_numpy(dtype=np.float64)
        ok = np.isfinite(rows)
        idx = np.flatnonzero(mask)
        idx = idx[ok]
        rows = rows[ok].astype(np.int64)
        owners = pos_mat[rows]
        scores = score_mat[rows]
        claimants_n[idx] = (owners >= 0).sum(axis=1).astype(np.float32)
        s1r = s1_row[idx].astype(np.int64)[:, None]
        hit = owners == s1r
        has = hit.any(axis=1)
        take = idx[has]
        if len(take) == 0:
            continue
        rank0 = hit[has].argmax(axis=1)
        ranks[take] = rank0.astype(np.float32) + 1.0
        sc = scores[has]
        cur = sc[np.arange(len(rank0)), rank0]
        nxt = np.zeros(len(rank0), dtype=np.float32)
        later = rank0 + 1 < sc.shape[1]
        nxt[later] = sc[np.arange(len(rank0))[later], rank0[later] + 1]
        gaps[take] = cur - nxt
    return {"reverse_rank": ranks, "reverse_gap": gaps, "reverse_claimants": claimants_n}


def matrix(matcher_p, packed_or_ranks, maxima, feats) -> np.ndarray:
    base_x = base.matrix(matcher_p, packed_or_ranks, maxima, feats)
    extra = np.column_stack([
        feats.get("reverse_rank", np.full(len(matcher_p), 99, np.float32)),
        feats.get("reverse_gap", np.zeros(len(matcher_p), np.float32)),
        feats.get("reverse_claimants", np.zeros(len(matcher_p), np.float32)),
    ])
    return np.column_stack([base_x, extra])


def train(x: np.ndarray, y: np.ndarray, path: Path | None = None):
    import lightgbm as lgb

    dataset = lgb.Dataset(x, label=y, feature_name=list(FEATURE_NAMES))
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": config.SEED,
    }
    model = lgb.train(params, dataset, num_boost_round=250)
    path = path or (STACKER_DIR / "model.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    (path.parent / "features.json").write_text(json.dumps(list(FEATURE_NAMES)))
    return model


def load(path: Path | None = None):
    import lightgbm as lgb

    path = path or (STACKER_DIR / "model.txt")
    return lgb.Booster(model_file=str(path))


def predict(model, x: np.ndarray) -> np.ndarray:
    return model.predict(x).astype(np.float64)
