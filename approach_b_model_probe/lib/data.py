"""Read the challenge TSVs and assign every Source 1 entity to a split.

Files are tab-separated because addresses and id lists contain commas.
Country is kept as an open string label; nothing here filters by it.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd

import config

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def source_path(split: str, source: int):
    return config.DATASET_DIR / split / f"{split}_source{source}.tsv"


def read_source(split: str, source: int) -> pd.DataFrame:
    frame = pd.read_csv(source_path(split, source), sep="\t", dtype=str, keep_default_na=False)
    missing = [column for column in SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{split} source {source} is missing columns {missing}")
    return frame[SOURCE_COLUMNS]


def parse_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()] if value else []


def read_truth(split: str = "train") -> dict[str, list[str]]:
    path = config.DATASET_DIR / split / f"{split}_ground_truth.tsv"
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {row.source1_entity_id: parse_ids(row.matched_entity_ids) for row in frame.itertuples(index=False)}


def _hash(entity_id: str, salt: str) -> int:
    digest = hashlib.blake2b(f"{salt}:{entity_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def split_of(entity_id: str) -> str:
    """90% train pool; the 10% hold-out is halved into calibration and evaluation."""
    if _hash(entity_id, "holdout") % config.HOLDOUT_MODULUS != 0:
        return "train"
    return "calibration" if _hash(entity_id, "half") % 2 == 0 else "evaluation"


def _match_bucket(count: int) -> str:
    if count <= 1:
        return str(count)
    if count <= 3:
        return "2-3"
    return "4-6" if count <= 6 else "7+"


def choose_entities(ids: np.ndarray, countries: np.ndarray, truth: dict[str, list[str]], size: int) -> dict[str, np.ndarray]:
    """Row positions for the training sample (stratified) and both hold-out halves."""
    splits = np.array([split_of(entity_id) for entity_id in ids])
    pool = np.flatnonzero(splits == "train")
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for position in pool:
        strata[(countries[position], _match_bucket(len(truth.get(ids[position], []))))].append(position)
    rng = np.random.default_rng(config.SEED)
    chosen: list[int] = []
    for members in strata.values():
        take = min(len(members), round(len(members) / len(pool) * size))
        chosen.extend(rng.choice(members, size=take, replace=False).tolist())
    return {
        "train": np.sort(np.array(chosen, dtype=np.int64)),
        "calibration": np.flatnonzero(splits == "calibration"),
        "evaluation": np.flatnonzero(splits == "evaluation"),
    }
