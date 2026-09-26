"""Global Source-1 claimant counts for a candidate name or address key.

A name claimant is a Source 1 record whose content tokens contain every
content token of the candidate (the candidate name is a token-subset).
An address claimant shares house number plus street core.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

import text


def _addr_key(house: str, street: str) -> str:
    house = str(house or "").lstrip("0")
    if house == "" and str(house or ""):
        house = "0"
    street = str(street or "").strip().lower()
    if not house or not street:
        return ""
    return f"{house}|{street}"


class ClaimantIndex:
    def __init__(self, frame: pd.DataFrame):
        names = (
            frame.name_norm.to_numpy() if "name_norm" in frame.columns else frame.name_roman.to_numpy()
        )
        if "house" in frame.columns:
            houses = frame.house.fillna("").astype(str).to_numpy()
            streets = frame.street_core.fillna("").astype(str).to_numpy()
        else:
            parsed = [text.parse_address(str(a), text.SEED_ALIASES) for a in frame.address_roman.to_numpy()]
            houses = np.array([row["house"] for row in parsed])
            streets = np.array([row["street_core"] for row in parsed])
        buckets: dict[str, list[int]] = defaultdict(list)
        for index, name in enumerate(names):
            for token in set(text.content_tokens(str(name))):
                buckets[token].append(index)
        self.name_post = {token: np.array(rows, dtype=np.int32) for token, rows in buckets.items()}
        addr_buckets: dict[str, list[int]] = defaultdict(list)
        for index, (house, street) in enumerate(zip(houses, streets)):
            key = _addr_key(house, street)
            if key:
                addr_buckets[key].append(index)
        self.addr_post = {key: np.array(rows, dtype=np.int32) for key, rows in addr_buckets.items()}
        self._name_cache: dict[str, int] = {}
        self._addr_cache: dict[str, int] = {}

    def name_count(self, name: str) -> int:
        cached = self._name_cache.get(name)
        if cached is not None:
            return cached
        tokens = sorted(set(text.content_tokens(name)), key=lambda tok: len(self.name_post.get(tok, ())))
        if not tokens or tokens[0] not in self.name_post:
            self._name_cache[name] = 0
            return 0
        current = self.name_post[tokens[0]]
        for token in tokens[1:]:
            other = self.name_post.get(token)
            if other is None:
                self._name_cache[name] = 0
                return 0
            current = np.intersect1d(current, other, assume_unique=True)
            if current.size == 0:
                self._name_cache[name] = 0
                return 0
        self._name_cache[name] = int(current.size)
        return self._name_cache[name]

    def address_count(self, house: str, street: str) -> int:
        key = _addr_key(house, street)
        if not key:
            return 0
        if key not in self._addr_cache:
            self._addr_cache[key] = int(len(self.addr_post.get(key, ())))
        return self._addr_cache[key]


def _series(frame: pd.DataFrame, column: str, default: str = "") -> np.ndarray:
    if column in frame.columns:
        return frame[column].fillna(default).astype(str).to_numpy()
    return np.full(len(frame), default, dtype=object)


def pair_features(
    s1_ids: np.ndarray,
    cand_ids: np.ndarray,
    s1: pd.DataFrame,
    others: dict[str, pd.DataFrame],
    index: ClaimantIndex,
) -> dict[str, np.ndarray]:
    """Vectorised field lookup; claimant counts are cached per unique candidate name/key."""
    s1_pos = {i: p for p, i in enumerate(s1.id.to_numpy())}
    other_pos = {
        key: {i: p for p, i in enumerate(frame.id.to_numpy())} for key, frame in others.items()
    }
    s1_idx = np.fromiter((s1_pos[i] for i in s1_ids), dtype=np.int32, count=len(s1_ids))
    cand_idx = np.empty(len(cand_ids), dtype=np.int32)
    source = np.empty(len(cand_ids), dtype=np.int8)
    for i, cand_id in enumerate(cand_ids):
        src = "S2" if str(cand_id).startswith("S2-") else "S3"
        source[i] = 2 if src == "S2" else 3
        cand_idx[i] = other_pos[src][cand_id]

    s1_names = _series(s1, "name_norm") if "name_norm" in s1.columns else _series(s1, "name_roman")
    s1_house = _series(s1, "house")
    s2n = _series(others["S2"], "name_norm") if "name_norm" in others["S2"].columns else _series(others["S2"], "name_roman")
    s3n = _series(others["S3"], "name_norm") if "name_norm" in others["S3"].columns else _series(others["S3"], "name_roman")
    s2a = _series(others["S2"], "address")
    s3a = _series(others["S3"], "address")
    s2h = _series(others["S2"], "house")
    s3h = _series(others["S3"], "house")
    s2s = _series(others["S2"], "street_core")
    s3s = _series(others["S3"], "street_core")
    s2nonce = _series(others["S2"], "nonce_like", "0")
    s3nonce = _series(others["S3"], "nonce_like", "0")
    s2dom = _series(others["S2"], "domain_like", "0")
    s3dom = _series(others["S3"], "domain_like", "0")

    n = len(cand_ids)
    r_name = np.empty(n, dtype=object)
    l_name = s1_names[s1_idx]
    r_addr = np.empty(n, dtype=object)
    r_house = np.empty(n, dtype=object)
    r_street = np.empty(n, dtype=object)
    nonce = np.zeros(n, dtype=np.int8)
    domain = np.zeros(n, dtype=np.int8)
    mask2 = source == 2
    r_name[mask2] = s2n[cand_idx[mask2]]
    r_name[~mask2] = s3n[cand_idx[~mask2]]
    r_addr[mask2] = s2a[cand_idx[mask2]]
    r_addr[~mask2] = s3a[cand_idx[~mask2]]
    r_house[mask2] = s2h[cand_idx[mask2]]
    r_house[~mask2] = s3h[cand_idx[~mask2]]
    r_street[mask2] = s2s[cand_idx[mask2]]
    r_street[~mask2] = s3s[cand_idx[~mask2]]
    nonce[mask2] = (s2nonce[cand_idx[mask2]] == "1").astype(np.int8)
    nonce[~mask2] = (s3nonce[cand_idx[~mask2]] == "1").astype(np.int8)
    domain[mask2] = (s2dom[cand_idx[mask2]] == "1").astype(np.int8)
    domain[~mask2] = (s3dom[cand_idx[~mask2]] == "1").astype(np.int8)

    blank = np.array([not str(a).strip() for a in r_addr], dtype=np.int8)
    if "nonce_like" not in others["S2"].columns:
        nonce = np.array([text.is_nonce_name(str(n_)) for n_ in r_name], dtype=np.int8)
    if "domain_like" not in others["S2"].columns:
        domain = np.array([text.is_domain_name(str(n_)) for n_ in r_name], dtype=np.int8)

    prefix = np.zeros(n, dtype=np.int8)
    house_conflict = np.zeros(n, dtype=np.int8)
    l_house = s1_house[s1_idx]
    name_claim = np.zeros(n, dtype=np.int32)
    addr_claim = np.zeros(n, dtype=np.int32)
    unique_names, name_inv = np.unique(r_name.astype(str), return_inverse=True)
    unique_counts = np.array([index.name_count(name) for name in unique_names], dtype=np.int32)
    name_claim = unique_counts[name_inv]
    unique_keys: list[str] = []
    key_inv = np.empty(n, dtype=np.int32)
    key_map: dict[str, int] = {}
    for i in range(n):
        key = _addr_key(str(r_house[i]), str(r_street[i]))
        if key not in key_map:
            key_map[key] = len(unique_keys)
            unique_keys.append(key)
        key_inv[i] = key_map[key]
        lt, rt = set(text.content_tokens(str(l_name[i]))), set(text.content_tokens(str(r_name[i])))
        prefix[i] = int(bool(rt) and rt <= lt)
        lh, rh = str(l_house[i]), str(r_house[i])
        if lh and rh and lh != rh:
            house_conflict[i] = 1
        if i and i % 1_000_000 == 0:
            print(f"pair features {i}/{n}", flush=True)
    addr_counts = np.zeros(len(unique_keys), dtype=np.int32)
    for j, key in enumerate(unique_keys):
        if "|" in key:
            house, street = key.split("|", 1)
            addr_counts[j] = index.address_count(house, street)
    addr_claim = addr_counts[key_inv]
    return {
        "blank_address": blank,
        "nonce_like": nonce,
        "domain_like": domain,
        "name_prefix": prefix,
        "house_conflict": house_conflict,
        "name_claimants": name_claim,
        "addr_claimants": addr_claim,
        "source": source,
    }
