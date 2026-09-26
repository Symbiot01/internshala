"""Prototype stacker_v2 on a copy of matcher_logits plus reverse features.

Does not overwrite models/stacker or reports/04_decision.json.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import calibrate
import claimants
import config
import data
import decide
import stacker_v2


def load_lookups(split: str, s2, s3):
    out = {}
    for source, frame in ((2, s2), (3, s3)):
        path = config.CACHE_DIR / split / "v2" / f"reverse_lookup_s{source}.npz"
        payload = np.load(path)
        id_map = {i: p for p, i in enumerate(frame.id.to_numpy())}
        out[source] = (payload["positions"], payload["scores"], id_map)
    return out


def main() -> None:
    src = config.CACHE_DIR / "train" / "matcher_logits.npz"
    if not src.exists():
        raise SystemExit("matcher_logits.npz not ready — wait for 04")
    dest = config.CACHE_DIR / "train" / "v2" / "logits_old.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    payload = np.load(dest, allow_pickle=True)
    decision_path = config.REPORTS_DIR / "04_decision.json"
    if not decision_path.exists():
        raise SystemExit("04_decision.json not ready")
    decision = json.loads(decision_path.read_text())
    temperature = float(decision["temperature"])
    truth = data.read_truth("train")
    s1 = pd.read_parquet(config.CACHE_DIR / "train" / "source1.parquet")
    s2 = pd.read_parquet(config.CACHE_DIR / "train" / "source2.parquet")
    s3 = pd.read_parquet(config.CACHE_DIR / "train" / "source3.parquet")
    groups = np.load(config.CACHE_DIR / "train" / "entities.npz")
    calib_ids = s1.id.to_numpy()[groups["calibration"]]
    eval_ids = s1.id.to_numpy()[groups["evaluation"]]
    lookups = load_lookups("train", s2, s3)
    index = claimants.ClaimantIndex(s1)
    others = {"S2": s2, "S3": s3}

    calib_p = calibrate.probabilities(payload["calib_logit"], temperature)
    eval_p = calibrate.probabilities(payload["eval_logit"], temperature)
    calib_pack = decide.pack(payload["calib_s1"], payload["calib_cand"], calib_p, payload["calib_label"], truth)
    print("calib pair features", flush=True)
    calib_feats = claimants.pair_features(calib_pack["s1_ids"], calib_pack["cand_ids"], s1, others, index)
    print("calib reverse features", flush=True)
    calib_feats.update(stacker_v2.reverse_features(calib_pack["s1_ids"], calib_pack["cand_ids"], s1, lookups))
    x_cal = stacker_v2.matrix(calib_pack["scores"], calib_pack["ranks_source"], calib_pack["maxima"], calib_feats)
    print("train lightgbm", flush=True)
    model = stacker_v2.train(x_cal, calib_pack["labels"].astype(np.float32))
    calib_stack = stacker_v2.predict(model, x_cal)
    stacked_pack = decide.pack(calib_pack["s1_ids"], calib_pack["cand_ids"], calib_stack, calib_pack["labels"], truth)
    rule = decide.grid_search(stacked_pack)
    triples = decide.select(stacked_pack, rule["threshold"], rule["cap"], rule["alpha"], rule["gate"])
    assigned = decide.unique_mapping(triples)
    calib_summary = decide.evaluate_assignment(assigned, calib_ids, truth)

    eval_pack = decide.pack(payload["eval_s1"], payload["eval_cand"], eval_p, payload["eval_label"], truth)
    print("eval pair features", flush=True)
    eval_feats = claimants.pair_features(eval_pack["s1_ids"], eval_pack["cand_ids"], s1, others, index)
    print("eval reverse features", flush=True)
    eval_feats.update(stacker_v2.reverse_features(eval_pack["s1_ids"], eval_pack["cand_ids"], s1, lookups))
    x_eval = stacker_v2.matrix(eval_pack["scores"], eval_pack["ranks_source"], eval_pack["maxima"], eval_feats)
    eval_stack = stacker_v2.predict(model, x_eval)
    eval_stacked = decide.pack(eval_pack["s1_ids"], eval_pack["cand_ids"], eval_stack, eval_pack["labels"], truth)
    triples = decide.select(eval_stacked, rule["threshold"], rule["cap"], rule["alpha"], rule["gate"])
    assigned = decide.unique_mapping(triples)
    eval_summary = decide.evaluate_assignment(assigned, eval_ids, truth)
    report = {
        "baseline_eval_f05": 0.9793,
        "rule": rule,
        "calibration": calib_summary,
        "evaluation": eval_summary,
        "delta_vs_9793": eval_summary["macro_f05"] - 0.9793,
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "14_stacker_v2.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
