"""Build labeled pairs from v2 candidate files. Does not touch pairs_*.parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
import data
import pairs as base


def cand_path(split: str, source: int) -> Path:
    return config.CACHE_DIR / split / "v2" / f"candidates_v2_s{source}.npz"


def pair_path(group: str) -> Path:
    folder = config.CACHE_DIR / ("test" if group == "test" else "train") / "v2"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"pairs_{group}.parquet"


def build_pairs(split: str, group: str) -> pd.DataFrame:
    data_split = "train" if group != "test" else "test"
    start, end = base.query_offsets(data_split)[group]
    s1 = base.load_prepared(data_split, 1)
    query_rows = np.load(cand_path(data_split, 2))["rows"][start:end]
    s1_ids = s1.id.to_numpy()[query_rows]
    truth = data.read_truth("train") if group != "test" else {}
    frames = []
    for source in (2, 3):
        payload = np.load(cand_path(data_split, source))
        positions = payload["positions"][start:end]
        scores = payload["scores"][start:end]
        other_ids = base.load_prepared(data_split, source).id.to_numpy()
        valid = positions >= 0
        s1_rep = np.repeat(s1_ids, valid.sum(axis=1))
        cand_ids = other_ids[positions[valid]]
        rrf = scores[valid]
        if truth:
            gold = [set(truth.get(entity, [])) for entity in s1_ids]
            labels = np.array(
                [cid in gold[i] for i, cid in zip(np.repeat(np.arange(len(s1_ids)), valid.sum(axis=1)), cand_ids)],
                dtype=np.int8,
            )
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


def write_pairs(group: str) -> dict[str, int]:
    table = build_pairs("train" if group != "test" else "test", group)
    path = pair_path(group)
    table.to_parquet(path, index=False)
    return {"rows": int(len(table)), "positives": int(table.label.sum())}


def load_pairs(group: str) -> pd.DataFrame:
    return pd.read_parquet(pair_path(group))
