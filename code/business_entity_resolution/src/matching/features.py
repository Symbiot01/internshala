"""Pairwise similarity features. Inputs are already normalized strings."""

from __future__ import annotations

import re

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from src.textnorm import content_tokens

_DIGITS = re.compile(r"\d+")

FEATURE_NAMES = (
    "name_jaro",
    "name_ratio",
    "name_token_sort",
    "name_partial",
    "name_jaccard",
    "name_char3_jaccard",
    "name_idf_overlap",
    "name_acronym",
    "name_len_delta",
    "name_prefix4",
    "addr_jaro",
    "addr_ratio",
    "addr_token_sort",
    "addr_partial",
    "addr_jaccard",
    "addr_char3_jaccard",
    "addr_num_jaccard",
    "addr_digit_jaccard",
    "same_country",
    "from_source3",
    "block_cosine",
    "block_rank",
    "cosine_over_best",
    "gap_to_second",
    "n_high_cosine",
    "mean_idf_name",
)

CHEAP_NAMES = (
    "name_jaccard",
    "name_prefix4",
    "name_len_delta",
    "addr_jaccard",
    "addr_num_jaccard",
    "addr_digit_jaccard",
    "same_country",
    "from_source3",
    "block_cosine",
    "block_rank",
    "cosine_over_best",
    "gap_to_second",
    "n_high_cosine",
    "mean_idf_name",
)

CHEAP_INDEX = tuple(FEATURE_NAMES.index(name) for name in CHEAP_NAMES)
HIGH_COSINE = 0.35


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    return overlap / (len(left) + len(right) - overlap)


def _ngrams(text: str, size: int) -> set[str]:
    compact = text.replace(" ", "")
    if len(compact) < size:
        return {compact} if compact else set()
    return {compact[i : i + size] for i in range(len(compact) - size + 1)}


def _digit_ngrams(text: str) -> set[str]:
    digits = "".join(char for char in text if char.isdigit())
    return _ngrams(digits, 2)


def _idf_overlap(left: list[str], right: list[str], idf: dict[str, float], default: float) -> float:
    if not left or not right:
        return 0.0
    left_set, right_set = set(left), set(right)
    shared = left_set & right_set
    if not shared:
        return 0.0
    union = left_set | right_set
    denom = sum(idf.get(token, default) for token in union)
    if denom <= 0:
        return 0.0
    return sum(idf.get(token, default) for token in shared) / denom


def _acronym(tokens: list[str]) -> str:
    return "".join(token[0] for token in tokens if token)


def cheap_vector(
    left_name: str,
    left_address: str,
    left_country: str,
    right_name: str,
    right_address: str,
    right_country: str,
    from_source3: bool,
    cosine: float,
    rank: int,
    mean_idf: float,
) -> list[float]:
    left_name_tokens = content_tokens(left_name)
    right_name_tokens = content_tokens(right_name)
    return [
        _jaccard(set(left_name_tokens), set(right_name_tokens)),
        1.0 if left_name[:4] and left_name[:4] == right_name[:4] else 0.0,
        abs(len(left_name) - len(right_name)) / max(len(left_name), len(right_name), 1),
        _jaccard(set(content_tokens(left_address)), set(content_tokens(right_address))),
        _jaccard(set(_DIGITS.findall(left_address)), set(_DIGITS.findall(right_address))),
        _jaccard(_digit_ngrams(left_address), _digit_ngrams(right_address)),
        1.0 if left_country == right_country and left_country else 0.0,
        1.0 if from_source3 else 0.0,
        cosine,
        rank / 49.0,
        0.0,
        0.0,
        0.0,
        mean_idf,
    ]


def full_vector(
    left_name: str,
    left_address: str,
    left_country: str,
    right_name: str,
    right_address: str,
    right_country: str,
    from_source3: bool,
    cosine: float,
    rank: int,
    mean_idf: float,
    idf: dict[str, float],
    idf_default: float,
) -> list[float]:
    left_name_tokens = content_tokens(left_name)
    right_name_tokens = content_tokens(right_name)
    left_addr_tokens = content_tokens(left_address)
    right_addr_tokens = content_tokens(right_address)
    acronym = _acronym(left_name_tokens)
    right_compact = right_name.replace(" ", "")
    acronym_hit = 0.0
    if len(acronym) >= 2 and (acronym == _acronym(right_name_tokens) or acronym in right_compact):
        acronym_hit = 1.0
    return [
        JaroWinkler.normalized_similarity(left_name, right_name),
        fuzz.ratio(left_name, right_name) / 100.0,
        fuzz.token_sort_ratio(left_name, right_name) / 100.0,
        fuzz.partial_ratio(left_name, right_name) / 100.0,
        _jaccard(set(left_name_tokens), set(right_name_tokens)),
        _jaccard(_ngrams(left_name, 3), _ngrams(right_name, 3)),
        _idf_overlap(left_name_tokens, right_name_tokens, idf, idf_default),
        acronym_hit,
        abs(len(left_name) - len(right_name)) / max(len(left_name), len(right_name), 1),
        1.0 if left_name[:4] and left_name[:4] == right_name[:4] else 0.0,
        JaroWinkler.normalized_similarity(left_address, right_address),
        fuzz.ratio(left_address, right_address) / 100.0,
        fuzz.token_sort_ratio(left_address, right_address) / 100.0,
        fuzz.partial_ratio(left_address, right_address) / 100.0,
        _jaccard(set(left_addr_tokens), set(right_addr_tokens)),
        _jaccard(_ngrams(left_address, 3), _ngrams(right_address, 3)),
        _jaccard(set(_DIGITS.findall(left_address)), set(_DIGITS.findall(right_address))),
        _jaccard(_digit_ngrams(left_address), _digit_ngrams(right_address)),
        1.0 if left_country == right_country and left_country else 0.0,
        1.0 if from_source3 else 0.0,
        cosine,
        rank / 49.0,
        0.0,
        0.0,
        0.0,
        mean_idf,
    ]
