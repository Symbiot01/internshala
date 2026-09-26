"""Optional DPR-style fine-tune of multilingual-e5-small (Karpukhin et al., 2020).

Recipe: InfoNCE with in-batch negatives plus one TF-IDF hard negative,
temperature 0.05, positives drawn only from training-pool entities. The
fine-tune is kept only when fused hold-out recall rises — see 02_fuse reports.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

import config
import data
import dense
import pairs as pair_tables
import text


def _retrieval_text(frame) -> list[str]:
    return (frame.name_roman + ", " + frame.address_roman).str.strip(", ").tolist()


def build_examples(limit: int) -> list[tuple[str, str, str]]:
    """(query, positive, hard-negative) triples from training-entity lexical hits."""
    s1 = pair_tables.load_prepared("train", 1)
    groups = np.load(config.CACHE_DIR / "train" / "entities.npz")
    train_rows = groups["train"]
    truth = data.read_truth("train")
    s1_ids = s1.id.to_numpy()[train_rows]
    s1_text = np.array(_retrieval_text(s1.iloc[train_rows]), dtype=object)
    examples: list[tuple[str, str, str]] = []
    for source in (2, 3):
        other = pair_tables.load_prepared("train", source)
        other_text = np.array(_retrieval_text(other), dtype=object)
        other_ids = other.id.to_numpy()
        id_to_text = dict(zip(other_ids.tolist(), other_text.tolist()))
        lex = np.load(config.CACHE_DIR / "train" / f"lexical_s{source}.npz")["positions"]
        prefix = f"S{source}-"
        for i, entity in enumerate(s1_ids):
            gold = {mid for mid in truth.get(entity, []) if mid.startswith(prefix) and mid in id_to_text}
            if not gold:
                continue
            hard = ""
            for pos in lex[i]:
                if pos < 0:
                    continue
                cand = other_ids[pos]
                if cand not in gold:
                    hard = id_to_text[cand]
                    break
            if not hard:
                continue
            for mid in gold:
                examples.append((s1_text[i], id_to_text[mid], hard))
                if len(examples) >= limit:
                    return examples
    return examples


def encode_batch(texts: list[str], tokenizer, model) -> torch.Tensor:
    batch = tokenizer(
        [config.DENSE_PREFIX + item for item in texts],
        max_length=config.DENSE_MAX_LEN,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    hidden = model(**batch).last_hidden_state
    return F.normalize(dense.mean_pool(hidden, batch["attention_mask"]).float(), dim=-1)


def main() -> None:
    examples = build_examples(config.DPR_POSITIVES)
    print(f"dpr triples {len(examples)}", flush=True)
    tokenizer, model = dense.load_encoder(config.DENSE_MODEL)
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    batch_size = 32
    steps = 0
    running = 0.0
    start = time.time()
    rng = np.random.default_rng(config.SEED)
    order = rng.permutation(len(examples))
    for begin in range(0, len(order), batch_size):
        rows = order[begin : begin + batch_size]
        queries = [examples[i][0] for i in rows]
        positives = [examples[i][1] for i in rows]
        hards = [examples[i][2] for i in rows]
        query_vec = encode_batch(queries, tokenizer, model)
        pos_vec = encode_batch(positives, tokenizer, model)
        hard_vec = encode_batch(hards, tokenizer, model)
        # In-batch positives plus one hard negative per query (DPR).
        documents = torch.cat([pos_vec, hard_vec], dim=0)
        logits = query_vec @ documents.T / config.DPR_TEMPERATURE
        targets = torch.arange(len(rows), device="cuda")
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        steps += 1
        running += float(loss.item())
        if steps % 50 == 0:
            print(f"dpr step {steps} loss {running / steps:.4f}", flush=True)
    out = config.MODELS_DIR / "dpr_e5"
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    report = {"triples": len(examples), "steps": steps, "loss": running / max(steps, 1), "seconds": round(time.time() - start, 1)}
    (config.REPORTS_DIR / "03_train_retriever.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)


if __name__ == "__main__":
    main()
