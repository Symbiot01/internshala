"""Score evaluation pairs with the frozen MiniLM checkpoint if logits.npz was deleted."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import cross_encoder
import distributed
import frames_util
import pairs as pair_tables
import score_util


def main() -> None:
    dest = config.CACHE_DIR / "train" / "matcher_logits.npz"
    if dest.exists():
        print(json.dumps({"skipped": True, "path": str(dest)}), flush=True)
        return
    distributed.maybe_relaunch()
    rank, world, device = distributed.setup()
    raw = {f"s{s}": pair_tables.load_prepared("train", s).set_index("id", drop=False) for s in (1, 2, 3)}
    frames = {k: frames_util.strip_prepared(v) for k, v in raw.items()}
    idf = cross_encoder.address_idf([frames_util.strip_prepared(pair_tables.load_prepared("train", s)) for s in (1, 2, 3)])
    tokenizer, model = score_util.load_scorer(config.MODELS_DIR / "minilm", device)
    eval_pairs = pair_tables.load_pairs("evaluation")
    logits = score_util.score_table(model, tokenizer, eval_pairs, frames, idf, device, rank, world)
    if rank == 0:
        np.savez(
            dest,
            eval_s1=eval_pairs.s1_id.to_numpy(),
            eval_cand=eval_pairs.cand_id.to_numpy(),
            eval_label=eval_pairs.label.to_numpy(),
            eval_logit=logits,
        )
        report = {"rows": int(len(eval_pairs)), "path": str(dest)}
        (config.REPORTS_DIR / "00b_minilm_eval.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    distributed.cleanup()


if __name__ == "__main__":
    main()
