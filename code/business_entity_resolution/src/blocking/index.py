"""Sparse word-token TF-IDF blocking.

One combined name+address index per source, max_df=0.01, top 50.
Measured on a 1,500-entity gold sample against full Source 2 after
normalization: recall@50 = 0.9642. Country is never a filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

MAX_DF = 0.01
TOP_K = 50
QUERY_BATCH = 500


@dataclass
class Index:
    vectorizer: TfidfVectorizer
    matrix_t: sp.csr_matrix  # term x doc, for fast query matmuls
    ids: np.ndarray


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[^\W_]{2,}",
        min_df=2,
        max_df=MAX_DF,
        sublinear_tf=True,
        dtype=np.float32,
    )


def fit_index(texts: list[str], ids: np.ndarray) -> Index:
    vectorizer = _vectorizer()
    matrix = vectorizer.fit_transform(texts)
    return Index(vectorizer, matrix.T.tocsr(), np.asarray(ids))


def query_topk(
    index: Index,
    queries: list[str],
    k: int = TOP_K,
    batch: int = QUERY_BATCH,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return per-query (doc positions, scores), best-first, for top-k each."""
    all_idx: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for start in range(0, len(queries), batch):
        sims = index.vectorizer.transform(queries[start : start + batch]) @ index.matrix_t
        sims = sims.tocsr()
        for i in range(sims.shape[0]):
            row = sims.getrow(i)
            if row.nnz == 0:
                all_idx.append(np.empty(0, dtype=np.int64))
                all_scores.append(np.empty(0, dtype=np.float32))
                continue
            if row.nnz > k:
                part = np.argpartition(-row.data, k - 1)[:k]
                order = part[np.argsort(-row.data[part])]
            else:
                order = np.argsort(-row.data)
            all_idx.append(row.indices[order].astype(np.int64))
            all_scores.append(row.data[order].astype(np.float32))
    return all_idx, all_scores


def combined_text(name: str, address: str) -> str:
    return f"{name} {address}".strip()
