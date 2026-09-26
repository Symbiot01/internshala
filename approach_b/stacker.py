"""LightGBM stacker on matcher probability plus claimant / flag features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config

FEATURE_NAMES = (
    "matcher_p",
    "rank_source",
    "gap_to_best",
    "blank_address",
    "nonce_like",
    "domain_like",
    "name_prefix",
    "house_conflict",
    "name_claimants",
    "addr_claimants",
    "source",
    "log_name_claimants",
    "log_addr_claimants",
)


def matrix(matcher_p: np.ndarray, packed_or_ranks: np.ndarray, maxima: np.ndarray, feats: dict[str, np.ndarray]) -> np.ndarray:
    name_c = feats["name_claimants"].astype(np.float32)
    addr_c = feats["addr_claimants"].astype(np.float32)
    gap = maxima - matcher_p
    cols = [
        matcher_p.astype(np.float32),
        packed_or_ranks.astype(np.float32),
        gap.astype(np.float32),
        feats["blank_address"].astype(np.float32),
        feats["nonce_like"].astype(np.float32),
        feats["domain_like"].astype(np.float32),
        feats["name_prefix"].astype(np.float32),
        feats["house_conflict"].astype(np.float32),
        name_c,
        addr_c,
        feats["source"].astype(np.float32),
        np.log1p(name_c),
        np.log1p(addr_c),
    ]
    return np.column_stack(cols)


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
    path = path or (config.STACKER_DIR / "model.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    (path.parent / "features.json").write_text(json.dumps(list(FEATURE_NAMES)))
    return model


def load(path: Path | None = None):
    import lightgbm as lgb

    path = path or (config.STACKER_DIR / "model.txt")
    return lgb.Booster(model_file=str(path))


def predict(model, x: np.ndarray) -> np.ndarray:
    return model.predict(x).astype(np.float64)
