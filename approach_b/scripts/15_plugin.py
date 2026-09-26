"""Score only new v2 pairs with the frozen matcher, retrain stacker_v2, re-grid.

Writes reports/15_decision.json. Does not overwrite 04_decision.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import calibrate
import claimants
import config
import cross_encoder
import data
import decide
import distributed
import pairs as old_pairs
import pairs_v2
import stacker_v2


def score_new(table: pd.DataFrame, frames, idf, include_country, device, rank, world, group: str) -> np.ndarray:
    tokenizer, model = cross_encoder.load_checkpoint(config.MODELS_DIR / "matcher", device=device)
    n = len(table)
    mine = np.flatnonzero(np.arange(n) % world == rank)
    shard_dir = config.CACHE_DIR / "train" / "v2" / f"new_logit_{group}_r{rank}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    step = 50_000
    chunks = []
    for start in range(0, len(mine), step):
        path = shard_dir / f"{start:08d}.npy"
        if path.exists():
            part = np.load(path)
            chunks.append(part)
            print(group, "rank", rank, "reuse", min(start + step, len(mine)), "/", len(mine), flush=True)
            continue
        rows = mine[start : start + step]
        left, right = cross_encoder.pair_texts(table.iloc[rows], frames, include_country=include_country, idf=idf)
        part = cross_encoder.predict_logits(model, tokenizer, left, right, device=device)
        np.save(path, part)
        chunks.append(part)
        print(group, "rank", rank, f"{min(start + step, len(mine))}/{len(mine)}", flush=True)
    part = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    distributed.barrier()
    full = config.CACHE_DIR / "train" / "v2" / f"new_logit_{group}_part_{rank}.npy"
    np.save(full, part)
    distributed.barrier()
    if rank != 0:
        return np.zeros(n, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    for r in range(world):
        out[np.arange(n) % world == r] = np.load(config.CACHE_DIR / "train" / "v2" / f"new_logit_{group}_part_{r}.npy")
    return out


def merge_logits(table: pd.DataFrame, new_logits: np.ndarray, old_s1, old_cand, old_logit):
    old = pd.DataFrame({"s1_id": old_s1, "cand_id": old_cand, "logit": old_logit})
    joined = table.merge(old, on=["s1_id", "cand_id"], how="left")
    missing = joined.logit.isna().to_numpy()
    merged = joined.logit.to_numpy(dtype=np.float32, copy=True)
    merged[missing] = new_logits
    return merged, int(missing.sum())


def assignment(s1, cand, scores, labels, truth, rule, eval_ids):
    packed = decide.pack(s1, cand, scores, labels, truth)
    triples = decide.select(packed, rule["threshold"], rule["cap"], rule["alpha"], rule["gate"])
    assigned = decide.unique_mapping(triples)
    return decide.evaluate_assignment(assigned, eval_ids, truth)


def main() -> None:
    distributed.maybe_relaunch()
    rank, world, device = distributed.setup()
    for group in ("calibration", "evaluation"):
        if rank == 0 and not pairs_v2.pair_path(group).exists():
            print("write", group, pairs_v2.write_pairs(group), flush=True)
    distributed.barrier()
    old_payload = np.load(config.CACHE_DIR / "train" / "matcher_logits.npz", allow_pickle=True)
    decision = json.loads((config.REPORTS_DIR / "04_decision.json").read_text())
    temperature = float(decision["temperature"])
    include_country = json.loads((config.REPORTS_DIR / "04_backbone.json").read_text()).get("include_country", True)
    frames = cross_encoder.lookup_frames("train")
    idf = cross_encoder.address_idf([old_pairs.load_prepared("train", source) for source in (1, 2, 3)])
    truth = data.read_truth("train")
    s1 = old_pairs.load_prepared("train", 1)
    s2 = old_pairs.load_prepared("train", 2)
    s3 = old_pairs.load_prepared("train", 3)
    groups = np.load(config.CACHE_DIR / "train" / "entities.npz")
    calib_ids = s1.id.to_numpy()[groups["calibration"]]
    eval_ids = s1.id.to_numpy()[groups["evaluation"]]
    lookups = {}
    for source, frame in ((2, s2), (3, s3)):
        payload = np.load(config.CACHE_DIR / "train" / "v2" / f"reverse_lookup_s{source}.npz")
        lookups[source] = (payload["positions"], payload["scores"], {i: p for p, i in enumerate(frame.id.to_numpy())})

    merged = {}
    new_counts = {}
    for group in ("calibration", "evaluation"):
        table = pairs_v2.load_pairs(group)
        prefix = "calib" if group == "calibration" else "eval"
        old = pd.DataFrame({
            "s1_id": old_payload[f"{prefix}_s1"],
            "cand_id": old_payload[f"{prefix}_cand"],
        })
        old["_old"] = np.int8(1)
        flag = table.merge(old, on=["s1_id", "cand_id"], how="left")["_old"].isna().to_numpy()
        new_table = table.loc[flag].reset_index(drop=True)
        print(group, "new_pairs", len(new_table), "of", len(table), flush=True)
        new_logits = (
            score_new(new_table, frames, idf, include_country, device, rank, world, group)
            if len(new_table)
            else np.zeros(0, np.float32)
        )
        if rank != 0:
            continue
        logits, n_new = merge_logits(
            table,
            new_logits,
            old_payload[f"{prefix}_s1"],
            old_payload[f"{prefix}_cand"],
            old_payload[f"{prefix}_logit"],
        )
        merged[group] = (table, logits)
        new_counts[group] = n_new
    if rank != 0:
        distributed.cleanup()
        return

    index = claimants.ClaimantIndex(s1)
    others = {"S2": s2, "S3": s3}
    calib_table, calib_logits = merged["calibration"]
    eval_table, eval_logits = merged["evaluation"]
    calib_p = calibrate.probabilities(calib_logits, temperature)
    eval_p = calibrate.probabilities(eval_logits, temperature)
    calib_pack = decide.pack(calib_table.s1_id.to_numpy(), calib_table.cand_id.to_numpy(), calib_p, calib_table.label.to_numpy(), truth)
    calib_feats = claimants.pair_features(calib_pack["s1_ids"], calib_pack["cand_ids"], s1, others, index)
    calib_feats.update(stacker_v2.reverse_features(calib_pack["s1_ids"], calib_pack["cand_ids"], s1, lookups))
    x_cal = stacker_v2.matrix(calib_pack["scores"], calib_pack["ranks_source"], calib_pack["maxima"], calib_feats)
    model = stacker_v2.train(x_cal, calib_pack["labels"].astype(np.float32))
    calib_stack = stacker_v2.predict(model, x_cal)
    stacked = decide.pack(calib_pack["s1_ids"], calib_pack["cand_ids"], calib_stack, calib_pack["labels"], truth)
    rule = decide.grid_search(stacked)
    calib_summary = assignment(calib_pack["s1_ids"], calib_pack["cand_ids"], calib_stack, calib_pack["labels"], truth, rule, calib_ids)

    eval_pack = decide.pack(eval_table.s1_id.to_numpy(), eval_table.cand_id.to_numpy(), eval_p, eval_table.label.to_numpy(), truth)
    eval_feats = claimants.pair_features(eval_pack["s1_ids"], eval_pack["cand_ids"], s1, others, index)
    eval_feats.update(stacker_v2.reverse_features(eval_pack["s1_ids"], eval_pack["cand_ids"], s1, lookups))
    x_eval = stacker_v2.matrix(eval_pack["scores"], eval_pack["ranks_source"], eval_pack["maxima"], eval_feats)
    eval_stack = stacker_v2.predict(model, x_eval)
    eval_summary = assignment(eval_pack["s1_ids"], eval_pack["cand_ids"], eval_stack, eval_pack["labels"], truth, rule, eval_ids)
    baseline = decision.get("evaluation", {}).get("macro_f05") or decision.get("phase1", {}).get("evaluation", {}).get("macro_f05", 0.9793)
    gate = {
        "eval_f05": eval_summary["macro_f05"],
        "baseline": baseline,
        "precision": eval_summary.get("macro_precision", 0),
        "singleton": eval_summary.get("singleton_accuracy", 0),
        "pass": bool(
            eval_summary["macro_f05"] >= baseline + 0.001
            and eval_summary.get("macro_precision", 0) >= 0.993
            and eval_summary.get("singleton_accuracy", 0) >= 0.978
        ),
    }
    report = {
        "temperature": temperature,
        "rule": rule,
        "calibration": calib_summary,
        "evaluation": eval_summary,
        "new_pairs": new_counts,
        "gate": gate,
        "score_source": "stacker_v2" if gate["pass"] else "keep_04",
    }
    (config.REPORTS_DIR / "15_decision.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str), flush=True)
    distributed.cleanup()


if __name__ == "__main__":
    main()
