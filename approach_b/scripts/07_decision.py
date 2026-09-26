"""Phase 1: re-grid the decision rule and optionally train the LightGBM stacker.

Uses existing matcher_logits.npz. No GPU.
"""

from __future__ import annotations

import argparse
import json
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
import stacker


def write_report(name: str, payload: dict) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str), flush=True)


def load_logits():
    payload = np.load(config.CACHE_DIR / "train" / "matcher_logits.npz", allow_pickle=True)
    decision = json.loads((config.REPORTS_DIR / "04_decision.json").read_text())
    temperature = float(decision["temperature"])
    return payload, temperature, decision


def assignment(s1, cand, scores, labels, truth, rule, eval_ids):
    packed = decide.pack(s1, cand, scores, labels, truth)
    triples = decide.select(packed, rule["threshold"], rule["cap"], rule["alpha"], rule["gate"])
    assigned = decide.unique_mapping(triples)
    summary = decide.evaluate_assignment(assigned, eval_ids, truth)
    return assigned, summary, packed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-stacker", action="store_true")
    args = parser.parse_args()
    payload, temperature, previous = load_logits()
    truth = data.read_truth("train")
    s1 = pd.read_parquet(config.CACHE_DIR / "train" / "source1.parquet")
    groups = np.load(config.CACHE_DIR / "train" / "entities.npz")
    calib_ids = s1.id.to_numpy()[groups["calibration"]]
    eval_ids = s1.id.to_numpy()[groups["evaluation"]]

    calib_p = calibrate.probabilities(payload["calib_logit"], temperature)
    eval_p = calibrate.probabilities(payload["eval_logit"], temperature)
    calib_packed = decide.pack(payload["calib_s1"], payload["calib_cand"], calib_p, payload["calib_label"], truth)
    rule = decide.grid_search(calib_packed)
    _, calib_summary, _ = assignment(
        payload["calib_s1"], payload["calib_cand"], calib_p, payload["calib_label"], truth, rule, calib_ids
    )
    assigned, eval_summary, _ = assignment(
        payload["eval_s1"], payload["eval_cand"], eval_p, payload["eval_label"], truth, rule, eval_ids
    )
    report = {
        "temperature": temperature,
        "rule": rule,
        "calibration": calib_summary,
        "evaluation": eval_summary,
        "previous_eval_f05": previous["evaluation"]["macro_f05"],
        "stacker": None,
    }

    if not args.skip_stacker:
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            print("lightgbm missing; installing is required for the stacker", flush=True)
            write_report("07_decision", report)
            return
        s2 = pd.read_parquet(config.CACHE_DIR / "train" / "source2.parquet")
        s3 = pd.read_parquet(config.CACHE_DIR / "train" / "source3.parquet")
        index = claimants.ClaimantIndex(s1)
        others = {"S2": s2, "S3": s3}
        print("building calib features", flush=True)
        calib_pack = decide.pack(payload["calib_s1"], payload["calib_cand"], calib_p, payload["calib_label"], truth)
        calib_feats = claimants.pair_features(calib_pack["s1_ids"], calib_pack["cand_ids"], s1, others, index)
        x_cal = stacker.matrix(calib_pack["scores"], calib_pack["ranks_source"], calib_pack["maxima"], calib_feats)
        model = stacker.train(x_cal, calib_pack["labels"].astype(np.float32))
        calib_stack = stacker.predict(model, x_cal)
        stacked_pack = decide.pack(calib_pack["s1_ids"], calib_pack["cand_ids"], calib_stack, calib_pack["labels"], truth)
        stacked_rule = decide.grid_search(stacked_pack)
        _, stacked_calib, _ = assignment(
            calib_pack["s1_ids"], calib_pack["cand_ids"], calib_stack, calib_pack["labels"], truth, stacked_rule, calib_ids
        )
        print("building eval features", flush=True)
        eval_pack = decide.pack(payload["eval_s1"], payload["eval_cand"], eval_p, payload["eval_label"], truth)
        eval_feats = claimants.pair_features(eval_pack["s1_ids"], eval_pack["cand_ids"], s1, others, index)
        x_eval = stacker.matrix(eval_pack["scores"], eval_pack["ranks_source"], eval_pack["maxima"], eval_feats)
        eval_stack = stacker.predict(model, x_eval)
        _, stacked_eval, _ = assignment(
            eval_pack["s1_ids"], eval_pack["cand_ids"], eval_stack, eval_pack["labels"], truth, stacked_rule, eval_ids
        )
        report["stacker"] = {
            "rule": stacked_rule,
            "calibration": stacked_calib,
            "evaluation": stacked_eval,
        }
        if stacked_eval["macro_f05"] > eval_summary["macro_f05"]:
            report["rule"] = stacked_rule
            report["calibration"] = stacked_calib
            report["evaluation"] = stacked_eval
            report["score_source"] = "stacker"
        else:
            report["score_source"] = "matcher"
            (config.STACKER_DIR / "disabled.json").write_text(json.dumps({"reason": "stacker lost to matcher on eval"}))
    write_report("07_decision", report)
    ship = report["evaluation"]["macro_f05"] >= config.SHIP_BASELINE + config.SHIP_MARGIN
    previous_path = config.REPORTS_DIR / "04_decision.json"
    merged = json.loads(previous_path.read_text())
    merged["phase1"] = report
    merged["rule"] = report["rule"]
    merged["evaluation"] = report["evaluation"]
    merged["calibration"] = report["calibration"]
    merged["ship"] = ship
    if report.get("score_source") == "stacker":
        merged["stacker"] = True
    previous_path.write_text(json.dumps(merged, indent=2, default=str))


if __name__ == "__main__":
    main()
