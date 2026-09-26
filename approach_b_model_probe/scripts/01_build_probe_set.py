"""Stratified MiniLM failure/control pairs plus oracle blocking-miss pairs."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import calibrate
import config
import data
import decide
import probe_rule
import text

INDIC = re.compile(r"[\u0900-\u0d7f]")
DOM = re.compile(r"www\.|\.com\b|\.in\b|\.net\b|\.org\b|\.fr\b")


def kind(s1_name: str, other_name: str, other_addr: str) -> str:
    if INDIC.search(other_name or "") or INDIC.search(other_addr or ""):
        return "indic"
    if DOM.search((other_name or "").lower()):
        return "domain"
    if not (other_addr or "").strip():
        return "blank_addr"
    if text.content_tokens(s1_name) and text.content_tokens(other_name):
        if not (set(text.content_tokens(s1_name)) & set(text.content_tokens(other_name))):
            return "nonce_alias"
    return "near_miss"


def _rec(frames, pos, entity_id: str) -> tuple[str, str]:
    table = frames[entity_id[:2]]
    row = pos[entity_id[:2]][entity_id]
    return table.name.iat[row], table.address.iat[row]


def _take(pool: pd.DataFrame, n: int, rng: np.random.Generator, kind_priority: bool) -> pd.DataFrame:
    if pool.empty or n <= 0:
        return pool.iloc[0:0]
    if kind_priority:
        order = ["blank_addr", "indic", "domain", "nonce_alias", "near_miss"]
        parts = []
        left = n
        for k in order:
            slice_ = pool[pool.kind == k]
            if slice_.empty or left <= 0:
                continue
            take = min(left, len(slice_))
            idx = rng.choice(len(slice_), size=take, replace=False)
            parts.append(slice_.iloc[idx])
            left -= take
        return pd.concat(parts, ignore_index=True) if parts else pool.iloc[0:0]
    take = min(n, len(pool))
    idx = rng.choice(len(pool), size=take, replace=False)
    return pool.iloc[idx].reset_index(drop=True)


def main() -> None:
    cache = config.CACHE_DIR / "train"
    s1 = pd.read_parquet(cache / "source1.parquet")
    s2 = pd.read_parquet(cache / "source2.parquet")
    s3 = pd.read_parquet(cache / "source3.parquet")
    frames = {"S1": s1, "S2": s2, "S3": s3}
    pos = {key: dict(zip(frame.id.to_numpy(), range(len(frame)))) for key, frame in frames.items()}
    truth = data.read_truth("train")
    groups = np.load(cache / "entities.npz")
    eval_ids = s1.id.to_numpy()[groups["evaluation"]]
    logits = np.load(cache / "matcher_logits.npz", allow_pickle=True)
    decision = json.loads((config.REPORTS_DIR / "04_decision.json").read_text())
    temperature = float(decision["temperature"])
    rule = decision["rule"]
    p = calibrate.probabilities(logits["eval_logit"], temperature)
    packed = decide.pack(logits["eval_s1"], logits["eval_cand"], p, logits["eval_label"], truth)
    assigned = probe_rule.assign_v04(packed, rule)
    key_index = {(str(a), str(b)): i for i, (a, b) in enumerate(zip(packed["s1_ids"], packed["cand_ids"]))}
    eval_pairs = pd.read_parquet(cache / "pairs_evaluation.parquet")

    below, hits, fps, misses = [], [], [], []
    for entity in eval_ids:
        gold = truth.get(entity, [])
        pred = assigned.get(entity, [])
        pred_set = set(pred)
        gold_set = set(gold)
        s1n, _ = _rec(frames, pos, entity)
        for gid in gold:
            loc = key_index.get((entity, gid))
            on, oa = _rec(frames, pos, gid)
            row = {
                "s1_id": entity,
                "cand_id": gid,
                "source": np.int8(2 if gid.startswith("S2-") else 3),
                "label": np.int8(1),
                "kind": kind(s1n, on, oa),
            }
            if loc is None:
                row["bucket"] = "blocking_miss"
                misses.append(row)
            elif gid in pred_set:
                row["bucket"] = "hit"
                hits.append(row)
            elif packed["scores"][loc] < rule["threshold"]:
                row["bucket"] = "below_threshold"
                below.append(row)
            else:
                row["bucket"] = "other_fn"
                below.append(row)
        for cid in pred:
            if cid not in gold_set:
                on, oa = _rec(frames, pos, cid)
                fps.append(
                    {
                        "s1_id": entity,
                        "cand_id": cid,
                        "source": np.int8(2 if str(cid).startswith("S2-") else 3),
                        "label": np.int8(0),
                        "kind": kind(s1n, on, oa),
                        "bucket": "fp",
                    }
                )

    singleton_ids = [e for e in eval_ids if not truth.get(e, []) and not assigned.get(e, [])]
    rng = np.random.default_rng(config.SEED)
    by_s1 = {sid: grp for sid, grp in eval_pairs.groupby("s1_id", sort=False)}
    singleton_rows = []
    if singleton_ids:
        chosen = rng.choice(singleton_ids, size=min(config.PROBE_SINGLETONS, len(singleton_ids)), replace=False)
        for entity in chosen:
            cands = by_s1.get(entity)
            if cands is None or cands.empty:
                continue
            pick = cands.iloc[int(rng.integers(0, len(cands)))]
            singleton_rows.append(
                {
                    "s1_id": entity,
                    "cand_id": pick.cand_id,
                    "source": np.int8(pick.source),
                    "label": np.int8(0),
                    "kind": "singleton",
                    "bucket": "singleton",
                }
            )

    below_df = pd.DataFrame(below)
    hits_df = pd.DataFrame(hits)
    fps_df = pd.DataFrame(fps)
    miss_df = pd.DataFrame(misses)
    pieces = [
        _take(below_df, config.PROBE_BELOW, rng, True),
        _take(hits_df, config.PROBE_HITS, rng, False),
        _take(fps_df, config.PROBE_FPS, rng, False),
        pd.DataFrame(singleton_rows),
    ]
    pieces = [p for p in pieces if p is not None and len(p) and len(p.columns)]
    probe = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    probe_path = config.CACHE_DIR / "probe_pairs.parquet"
    probe.to_parquet(probe_path, index=False)

    oracle_rows = []
    if not miss_df.empty:
        take_m = min(config.ORACLE_GOLD, len(miss_df))
        gold_sample = miss_df.iloc[rng.choice(len(miss_df), size=take_m, replace=False)]
        for row in gold_sample.itertuples(index=False):
            oracle_rows.append(
                {
                    "s1_id": row.s1_id,
                    "cand_id": row.cand_id,
                    "source": np.int8(row.source),
                    "label": np.int8(1),
                    "kind": row.kind,
                    "bucket": "blocking_miss",
                }
            )
            neigh = by_s1.get(row.s1_id)
            if neigh is None:
                continue
            neigh = neigh[neigh.cand_id != row.cand_id]
            if neigh.empty:
                continue
            k = min(config.ORACLE_NEGS, len(neigh))
            negs = neigh.iloc[rng.choice(len(neigh), size=k, replace=False)]
            for neg in negs.itertuples(index=False):
                oracle_rows.append(
                    {
                        "s1_id": row.s1_id,
                        "cand_id": neg.cand_id,
                        "source": np.int8(neg.source),
                        "label": np.int8(0),
                        "kind": "oracle_neg",
                        "bucket": "oracle_neg",
                    }
                )
    oracle = pd.DataFrame(oracle_rows)
    oracle_path = config.CACHE_DIR / "oracle_pairs.parquet"
    oracle.to_parquet(oracle_path, index=False)

    report = {
        "probe_rows": int(len(probe)),
        "probe_by_bucket": probe.bucket.value_counts().to_dict(),
        "probe_by_kind": probe.kind.value_counts().to_dict(),
        "oracle_rows": int(len(oracle)),
        "pool": {
            "below": int(len(below_df)),
            "hits": int(len(hits_df)),
            "fps": int(len(fps_df)),
            "blocking_miss": int(len(miss_df)),
        },
        "rule": {k: rule[k] for k in ("threshold", "cap", "alpha", "gate")},
        "temperature": temperature,
    }
    (config.REPORTS_DIR / "01_probe_set.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
