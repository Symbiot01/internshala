"""Score probe/oracle pairs with MiniLM lookup and a new checkpoint; fit temperature."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import calibrate
import config
import cross_encoder
import distributed
import frames_util
import pairs as pair_tables
import score_util


def minilm_lookup(s1: np.ndarray, cand: np.ndarray) -> np.ndarray:
    payload = np.load(config.CACHE_DIR / "train" / "matcher_logits.npz", allow_pickle=True)
    index = {(str(a), str(b)): i for i, (a, b) in enumerate(zip(payload["eval_s1"], payload["eval_cand"]))}
    out = np.full(len(s1), np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(zip(s1, cand)):
        loc = index.get((str(a), str(b)))
        if loc is not None:
            out[i] = payload["eval_logit"][loc]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xlmr", "qwen3b"], default="xlmr")
    args = parser.parse_args()
    ckpt = config.MODELS_DIR / ("xlmr" if args.model == "xlmr" else "qwen3b_lora")
    if not ckpt.exists():
        raise SystemExit(f"missing checkpoint {ckpt}")
    distributed.maybe_relaunch()
    rank, world, device = distributed.setup()
    frames = cross_encoder.lookup_frames("train")
    frames = {k: frames_util.strip_prepared(v) for k, v in frames.items()}
    idf = cross_encoder.address_idf(
        [frames_util.strip_prepared(pair_tables.load_prepared("train", source)) for source in (1, 2, 3)]
    )
    tokenizer, model = score_util.load_scorer(ckpt, device)

    calib = pair_tables.load_pairs("calibration")
    if len(calib) > config.CALIB_T_PAIRS:
        calib = calib.sample(config.CALIB_T_PAIRS, random_state=config.SEED).reset_index(drop=True)
    calib_logits = score_util.score_table(model, tokenizer, calib, frames, idf, device, rank, world)
    probe = pd.read_parquet(config.CACHE_DIR / "probe_pairs.parquet")
    oracle = pd.read_parquet(config.CACHE_DIR / "oracle_pairs.parquet")
    probe_logits = score_util.score_table(model, tokenizer, probe, frames, idf, device, rank, world)
    oracle_logits = score_util.score_table(model, tokenizer, oracle, frames, idf, device, rank, world)
    if rank == 0:
        temperature = calibrate.fit_temperature(calib_logits, calib.label.to_numpy())
        probe = probe.copy()
        probe["logit_new"] = probe_logits
        probe["logit_minilm"] = minilm_lookup(probe.s1_id.to_numpy(), probe.cand_id.to_numpy())
        oracle = oracle.copy()
        oracle["logit_new"] = oracle_logits
        out_dir = config.CACHE_DIR / "scores"
        out_dir.mkdir(parents=True, exist_ok=True)
        probe.to_parquet(out_dir / f"probe_{args.model}.parquet", index=False)
        oracle.to_parquet(out_dir / f"oracle_{args.model}.parquet", index=False)
        report = {
            "model": args.model,
            "checkpoint": str(ckpt),
            "temperature": temperature,
            "calib_pairs": int(len(calib)),
            "probe_rows": int(len(probe)),
            "oracle_rows": int(len(oracle)),
            "probe_minilm_missing": int(np.isnan(probe.logit_minilm.to_numpy()).sum()),
        }
        (config.REPORTS_DIR / f"04_score_{args.model}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    distributed.cleanup()


if __name__ == "__main__":
    main()
