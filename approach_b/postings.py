"""Exact-key inverted indexes for house/street/city and other atom keys."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

import config


def build(keys: list[str], max_df: int | None = None) -> dict[str, np.ndarray]:
    max_df = config.POSTING_MAX_DF if max_df is None else max_df
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        if key:
            buckets[key].append(index)
    return {
        key: np.array(rows, dtype=np.int32)
        for key, rows in buckets.items()
        if 1 <= len(rows) <= max_df
    }


def search(query_keys: list[str], postings: dict[str, np.ndarray], k: int) -> np.ndarray:
    positions = np.full((len(query_keys), k), -1, dtype=np.int32)
    for row, key in enumerate(query_keys):
        docs = postings.get(key)
        if docs is None or len(docs) == 0:
            continue
        take = min(k, len(docs))
        positions[row, :take] = docs[:take]
    return positions


def search_union(query_key_lists: list[list[str]], postings: dict[str, np.ndarray], k: int) -> np.ndarray:
    positions = np.full((len(query_key_lists), k), -1, dtype=np.int32)
    for row, keys in enumerate(query_key_lists):
        seen: list[int] = []
        used: set[int] = set()
        for key in keys:
            docs = postings.get(key)
            if docs is None:
                continue
            for doc in docs:
                value = int(doc)
                if value not in used:
                    used.add(value)
                    seen.append(value)
                    if len(seen) >= k:
                        break
            if len(seen) >= k:
                break
        if seen:
            positions[row, :len(seen)] = seen
    return positions
