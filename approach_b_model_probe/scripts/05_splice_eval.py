"""Splice new logits into eval MiniLM scores and replay the 04 decision rule."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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


def replay(s1, cand, logits, labels, temperature, rule, truth, eval_ids):
    probs = calibrate.probabilities(logits, temperature)
    packed = decide.pack(s1, cand, probs, labels, truth)
    assigned = probe_rule.assign_v04(packed, rule)
    summary = decide.evaluate_assignment(assigned, eval_ids, truth)
    return assigned, summary, packed


def bucket_recovery(probe: pd.DataFrame, minilm_t: float, new_t: float, rule: dict) -> dict:
    out: dict[str, dict] = {}
    t = rule["threshold"]
    for name, frame in probe.groupby("bucket"):
        p0 = calibrate.probabilities(frame.logit_minilm.to_numpy(), minilm_t)
        p1 = calibrate.probabilities(frame.logit_new.to_numpy(), new_t)
        gold = frame.label.to_numpy() == 1
        neg = frame.label.to_numpy() == 0
        out[str(name)] = {
            "n": int(len(frame)),
            "gold": int(gold.sum()),
            "minilm_above_t": int(((p0 >= t) & gold).sum()),
            "new_above_t": int(((p1 >= t) & gold).sum()),
            "minilm_fp_above_t": int(((p0 >= t) & neg).sum()),
            "new_fp_above_t": int(((p1 >= t) & neg).sum()),
        }
    by_kind = defaultdict(lambda: {"n": 0, "gold": 0, "new_above_t": 0, "minilm_above_t": 0})
    t = rule["threshold"]
    p0 = calibrate.probabilities(probe.logit_minilm.to_numpy(), minilm_t)
    p1 = calibrate.probabilities(probe.logit_new.to_numpy(), new_t)
    gold = probe.label.to_numpy() == 1
    for i, k in enumerate(probe.kind.to_numpy()):
        by_kind[str(k)]["n"] += 1
        if gold[i]:
            by_kind[str(k)]["gold"] += 1
            if p0[i] >= t:
                by_kind[str(k)]["minilm_above_t"] += 1
            if p1[i] >= t:
                by_kind[str(k)]["new_above_t"] += 1
    out["by_kind"] = dict(by_kind)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xlmr", "qwen3b"], default="xlmr")
    args = parser.parse_args()
    score_path = config.CACHE_DIR / "scores" / f"probe_{args.model}.parquet"
    score_report = json.loads((config.REPORTS_DIR / f"04_score_{args.model}.json").read_text())
    new_t = float(score_report["temperature"])
    decision = json.loads((config.REPORTS_DIR / "04_decision.json").read_text())
    minilm_t = float(decision["temperature"])
    rule = decision["rule"]
    probe = pd.read_parquet(score_path)
    payload = np.load(config.CACHE_DIR / "train" / "matcher_logits.npz", allow_pickle=True)
    spliced = payload["eval_logit"].astype(np.float32).copy()
    index = {(str(a), str(b)): i for i, (a, b) in enumerate(zip(payload["eval_s1"], payload["eval_cand"]))}
    overwritten = 0
    for s1_id, cand_id, logit in zip(probe.s1_id, probe.cand_id, probe.logit_new):
        loc = index.get((str(s1_id), str(cand_id)))
        if loc is None:
            continue
        spliced[loc] = float(logit)
        overwritten += 1
    truth = data.read_truth("train")
    s1 = pd.read_parquet(config.CACHE_DIR / "train" / "source1.parquet")
    groups = np.load(config.CACHE_DIR / "train" / "entities.npz")
    eval_ids = s1.id.to_numpy()[groups["evaluation"]]
    _, base_summary, _ = replay(
        payload["eval_s1"], payload["eval_cand"], payload["eval_logit"], payload["eval_label"],
        minilm_t, rule, truth, eval_ids,
    )
    _, new_summary, _ = replay(
        payload["eval_s1"], payload["eval_cand"], spliced, payload["eval_label"],
        new_t, rule, truth, eval_ids,
    )
    d_f05 = new_summary["macro_f05"] - base_summary["macro_f05"]
    promote = d_f05 >= config.PROMOTE_D_F05 and new_summary["macro_precision"] >= config.PROMOTE_MIN_P
    report = {
        "model": args.model,
        "overwritten": overwritten,
        "minilm_temperature": minilm_t,
        "new_temperature": new_t,
        "rule": {k: rule[k] for k in ("threshold", "cap", "alpha", "gate")},
        "minilm": base_summary,
        "spliced": new_summary,
        "delta_macro_f05": d_f05,
        "promote": promote,
        "promote_gate": {"d_f05": config.PROMOTE_D_F05, "min_precision": config.PROMOTE_MIN_P},
        "probe_recovery": bucket_recovery(probe, minilm_t, new_t, rule),
    }
    oracle_path = config.CACHE_DIR / "scores" / f"oracle_{args.model}.parquet"
    if oracle_path.exists():
        oracle = pd.read_parquet(oracle_path)
        og = oracle[oracle.bucket == "blocking_miss"]
        if not og.empty:
            p1 = calibrate.probabilities(og.logit_new.to_numpy(), new_t)
            report["oracle_blocking_miss_new_above_t"] = float((p1 >= rule["threshold"]).mean())
    out_name = f"delta_{args.model}.json"
    (config.REPORTS_DIR / out_name).write_text(json.dumps(report, indent=2))
    if args.model == "xlmr":
        (config.REPORTS_DIR / "delta.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if args.model == "xlmr" and not promote:
        print("GATE FAIL: do not train 3B", flush=True)


if __name__ == "__main__":
    main()
