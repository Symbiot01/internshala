"""Word TF-IDF retrieval (Ditto's case study blocked on TF-IDF over name and address).

The index is fit on the split being searched, which is unsupervised and puts
test-only vocabulary (for example French street words) into the index.
Queries run in forked worker processes that share the index read-only.
"""

from __future__ import annotations

from functools import partial
from multiprocessing import get_context

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import config

QUERY_CHUNK = 1000
_SHARED_INDEX: tuple[TfidfVectorizer, object] | None = None


def build_index(documents: list[str]) -> tuple[TfidfVectorizer, object]:
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[^\W_]{2,}",
        min_df=config.LEXICAL_MIN_DF,
        max_df=config.LEXICAL_MAX_DF,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix.T.tocsr()  # term x document, so a query is one sparse matmul


def _search_chunk(queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
    vectorizer, term_by_doc = _SHARED_INDEX
    similarities = (vectorizer.transform(queries) @ term_by_doc).tocsr()
    positions = np.full((len(queries), k), -1, dtype=np.int32)
    scores = np.zeros((len(queries), k), dtype=np.float32)
    for row in range(similarities.shape[0]):
        start, end = similarities.indptr[row], similarities.indptr[row + 1]
        if start == end:
            continue
        values = similarities.data[start:end]
        columns = similarities.indices[start:end]
        take = min(k, len(values))
        best = np.argpartition(-values, take - 1)[:take]
        best = best[np.argsort(-values[best])]
        positions[row, :take] = columns[best]
        scores[row, :take] = values[best]
    return positions, scores


def search(index: tuple[TfidfVectorizer, object], queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k document positions (best first, -1 padded) and cosine scores for each query."""
    global _SHARED_INDEX
    _SHARED_INDEX = index
    chunks = [queries[i : i + QUERY_CHUNK] for i in range(0, len(queries), QUERY_CHUNK)]
    print(f"lexical search: {len(queries)} queries, {len(chunks)} chunks, {config.WORKERS} workers", flush=True)
    with get_context("fork").Pool(config.WORKERS) as pool:
        parts = pool.map(partial(_search_chunk, k=k), chunks)
    _SHARED_INDEX = None
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
