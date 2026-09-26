"""Reproduce hold-out gold-loss decomposition by type, country, and source."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import calibrate
import config
import data
import decide
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
    keep = decide.keep_mask(packed, rule["threshold"], rule["cap"], rule["alpha"], rule["gate"])
    triples = decide.select(packed, rule["threshold"], rule["cap"], rule["alpha"], rule["gate"])
    assigned = decide.unique_mapping(triples)
    claimed = {cand: s1_id for s1_id, cand, _ in sorted(triples, key=lambda row: -row[2])}
    key_index = {(str(a), str(b)): i for i, (a, b) in enumerate(zip(packed["s1_ids"], packed["cand_ids"]))}
    rec = lambda i: (
        frames[i[:2]].name.iat[pos[i[:2]][i]],
        frames[i[:2]].address.iat[pos[i[:2]][i]],
    )
    stats: Counter = Counter()
    by: dict[str, Counter] = defaultdict(Counter)
    for entity in eval_ids:
        for gid in truth.get(entity, []):
            loc = key_index.get((entity, gid))
            if loc is None:
                bucket = "blocking_miss"
            elif entity in assigned and gid in assigned.get(entity, []):
                bucket = "hit"
            elif packed["scores"][loc] < rule["threshold"]:
                bucket = "below_threshold"
            elif packed["ranks_source"][loc] > min(rule["cap"], config.SOURCE_CAP_S2 if str(gid).startswith("S2-") else config.SOURCE_CAP_S3):
                bucket = "cap_cut"
            else:
                bucket = "umc_or_alpha"
            stats[bucket] += 1
            s1n, _ = rec(entity)
            on, oa = rec(gid)
            k = kind(s1n, on, oa)
            country = s1.country.iat[pos["S1"][entity]]
            by[bucket][k] += 1
            by[bucket][country] += 1
            by[bucket][gid[:2]] += 1
    summary = decide.evaluate_assignment(assigned, eval_ids, truth)
    report = {
        "evaluation": summary,
        "gold_pairs": int(sum(stats.values())),
        "loss": dict(stats),
        "by_bucket": {k: dict(v) for k, v in by.items()},
        "rule": rule,
        "note": "keep_mask uses per-source ranks and cap-exempt from decide.py",
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "06_error_ledger.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
