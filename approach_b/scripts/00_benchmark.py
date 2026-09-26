"""Stage 0: fetch the three MIT models and measure throughput on this GPU.

Writes reports/00_benchmark.json. Every runtime decision later reads from it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

import config
import data
import text


def fetch_models() -> None:
    for repo in [config.DENSE_MODEL, config.TOKENIZER_NAME, *config.BACKBONES.values()]:
        snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model"])


def sample_records(split: str, source: int, rows: int) -> list[str]:
    frame = data.read_source(split, source).sample(rows, random_state=config.SEED)
    serialized = []
    for name, address, country in zip(frame.business_name, frame.business_address, frame.country):
        n, a, nr, ar = text.prepare_record(name, address)
        serialized.append(text.serialize(n, a, nr, ar, country, dk_tags=True))
    return serialized


def time_it(step, repeats: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    fetch_models()
    left = sample_records("train", 1, 20_000)
    right = sample_records("train", 2, 20_000)
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    tokenizer.add_tokens(config.SPECIAL_TOKENS, special_tokens=True)
    lengths = np.array([len(ids) for ids in tokenizer(left, right)["input_ids"]])
    length_report = {q: int(np.percentile(lengths, q)) for q in (50, 90, 99, 99.5, 99.9)}
    overflow = float((lengths > config.MAX_LEN).mean())
    print("pair token lengths", length_report, "overflow at", config.MAX_LEN, overflow, flush=True)

    report = {"pair_token_percentiles": length_report, "overflow_fraction_at_max_len": overflow}
    device = "cuda"

    encoder_tokenizer = AutoTokenizer.from_pretrained(config.DENSE_MODEL)
    encoder = AutoModel.from_pretrained(config.DENSE_MODEL, dtype=torch.bfloat16).to(device).eval()
    batch = encoder_tokenizer([config.DENSE_PREFIX + item for item in right[:2048]], max_length=config.DENSE_MAX_LEN,
                              truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        seconds = time_it(lambda: encoder(**batch), repeats=5)
    report["e5_small_texts_per_second"] = round(2048 / seconds)
    print("e5-small texts/s", report["e5_small_texts_per_second"], flush=True)
    del encoder

    pair_batch = tokenizer(left[:1024], right[:1024], max_length=config.MAX_LEN, truncation=True,
                           padding=True, return_tensors="pt").to(device)
    train_batch = {key: value[: config.BATCH_SIZE] for key, value in pair_batch.items()}
    for key, repo in config.BACKBONES.items():
        model = AutoModelForSequenceClassification.from_pretrained(repo, num_labels=1)
        model.resize_token_embeddings(max(len(tokenizer), model.config.vocab_size))
        model.to(device)
        model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            infer_seconds = time_it(lambda: model(**pair_batch), repeats=5)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        labels = torch.zeros(config.BATCH_SIZE, device=device)

        def train_step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**train_batch).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_step()
        train_seconds = time_it(train_step, repeats=10)
        report[f"{key}_predict_pairs_per_second"] = round(1024 / infer_seconds)
        report[f"{key}_train_pairs_per_second"] = round(config.BATCH_SIZE / train_seconds)
        print(key, "predict pairs/s", report[f"{key}_predict_pairs_per_second"],
              "train pairs/s", report[f"{key}_train_pairs_per_second"], flush=True)
        del model, optimizer
        torch.cuda.empty_cache()

    report["gpu"] = torch.cuda.get_device_name(0)
    report["torch"] = torch.__version__
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "00_benchmark.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
