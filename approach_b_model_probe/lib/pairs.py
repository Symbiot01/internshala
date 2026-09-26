"""Labeled candidate pairs for the cross-encoder.

A pair is a (Source 1 entity, retrieved Source 2/3 record). The label is 1 when
the pair is in the ground truth. Negatives are retrieval near-misses, so they
are hard by construction. The challenge ground truth is complete, so a retrieved
record that is not listed is a true negative — no denoising.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
import data


def load_prepared(split: str, source: int) -> pd.DataFrame:
    return pd.read_parquet(config.CACHE_DIR / split / f"source{source}.parquet")


def query_offsets(split: str) -> dict[str, tuple[int, int]]:
    """Row ranges inside the retrieval matrices (train queries are concatenated)."""
    if split == "test":
        n = len(load_prepared("test", 1))
        return {"test": (0, n)}
    groups = np.load(config.CACHE_DIR / "train" / "entities.npz")
    start = 0
    out: dict[str, tuple[int, int]] = {}
    for name in ("train", "calibration", "evaluation"):
        end = start + len(groups[name])
        out[name] = (start, end)
        start = end
    return out


def build_pairs(split: str, group: str) -> pd.DataFrame:
    """One row per retrieved candidate for the entities in ``group``."""
    data_split = "train" if group != "test" else "test"
    start, end = query_offsets(data_split)[group]
    s1 = load_prepared(data_split, 1)
    query_rows = np.load(config.CACHE_DIR / data_split / "candidates_s2.npz")["rows"][start:end]
    s1_ids = s1.id.to_numpy()[query_rows]
    truth = data.read_truth("train") if group != "test" else {}

    frames = []
    for source in (2, 3):
        payload = np.load(config.CACHE_DIR / data_split / f"candidates_s{source}.npz")
        positions = payload["positions"][start:end]
        scores = payload["scores"][start:end]
        other_ids = load_prepared(data_split, source).id.to_numpy()
        valid = positions >= 0
        s1_rep = np.repeat(s1_ids, valid.sum(axis=1))
        cand_ids = other_ids[positions[valid]]
        rrf = scores[valid]
        if truth:
            gold = [set(truth.get(entity, [])) for entity in s1_ids]
            labels = np.array([cid in gold[i] for i, cid in zip(np.repeat(np.arange(len(s1_ids)), valid.sum(axis=1)), cand_ids)], dtype=np.int8)
        else:
            labels = np.zeros(len(cand_ids), dtype=np.int8)
        frames.append(pd.DataFrame({
            "s1_id": s1_rep,
            "cand_id": cand_ids,
            "source": np.int8(source),
            "label": labels,
            "rrf": rrf.astype(np.float32),
        }))
    return pd.concat(frames, ignore_index=True)


def pair_path(group: str) -> Path:
    folder = config.CACHE_DIR / ("test" if group == "test" else "train")
    return folder / f"pairs_{group}.parquet"


def write_pairs(group: str) -> dict[str, int]:
    table = build_pairs("train" if group != "test" else "test", group)
    path = pair_path(group)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return {
        "rows": int(len(table)),
        "positives": int(table.label.sum()),
        "entities": int(table.s1_id.nunique()),
    }


def load_pairs(group: str) -> pd.DataFrame:
    return pd.read_parquet(pair_path(group))


def sample_train_entities(table: pd.DataFrame, size: int, seed: int = config.SEED) -> pd.DataFrame:
    """Keep every pair of ``size`` randomly chosen Source 1 ids (for the bake-off)."""
    ids = table.s1_id.drop_duplicates()
    take = ids.sample(n=min(size, len(ids)), random_state=seed)
    return table[table.s1_id.isin(take)].reset_index(drop=True)
