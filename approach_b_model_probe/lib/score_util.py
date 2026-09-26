"""Score a pair table with a sequence-classification checkpoint (DDP-aware)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
import cross_encoder
import distributed
import frames_util


def score_table(model, tokenizer, table: pd.DataFrame, frames, idf, device, rank: int, world: int) -> np.ndarray:
    frames = {key: frames_util.strip_prepared(value) for key, value in frames.items()}
    n = len(table)
    mine = table.iloc[np.arange(n) % world == rank].reset_index(drop=True)
    left, right = cross_encoder.pair_texts(
        mine, frames, include_country=config.INCLUDE_COUNTRY, idf=idf, attr_del_p=0.0, augment=False
    )
    part = cross_encoder.predict_logits(model, tokenizer, left, right, device=device)
    if world == 1:
        return part
    path = config.CACHE_DIR / f"score_part_{rank}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, part)
    distributed.barrier()
    if rank != 0:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    for r in range(world):
        out[np.arange(n) % world == r] = np.load(config.CACHE_DIR / f"score_part_{r}.npy")
    return out


def load_scorer(ckpt: Path, device):
    ckpt = Path(ckpt)
    if (ckpt / "adapter_config.json").exists():
        from peft import PeftModel

        tokenizer = cross_encoder._from_pretrained(cross_encoder.AutoTokenizer.from_pretrained, str(ckpt))
        meta = json.loads((ckpt / "probe_meta.json").read_text()) if (ckpt / "probe_meta.json").exists() else {}
        base_name = meta.get("base", config.BACKBONES["qwen3b"])
        base = cross_encoder._from_pretrained(
            cross_encoder.AutoModelForSequenceClassification.from_pretrained,
            base_name,
            num_labels=1,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            base.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base, str(ckpt))
        return tokenizer, model.to(device).eval()
    tokenizer, model = cross_encoder.load_checkpoint(ckpt, device=device)
    model.eval()
    return tokenizer, model
