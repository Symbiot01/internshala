"""Ditto-style sequence-pair matcher (Li et al., PVLDB 2020).

Each record is serialized as ``[COL] attr [VAL] value``. The model is a
multilingual encoder with a linear head, trained with binary cross-entropy
on retrieved pairs. Prediction is length-sorted bf16 inference.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from dense import _from_pretrained

import config
import distributed
import pairs as pair_tables
import text


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(name: str = config.TOKENIZER_NAME):
    tokenizer = _from_pretrained(AutoTokenizer.from_pretrained, name)
    tokenizer.add_tokens(config.SPECIAL_TOKENS, special_tokens=True)
    return tokenizer


def load_model(backbone: str, tokenizer, device: torch.device | str = "cuda") -> torch.nn.Module:
    model = _from_pretrained(
        AutoModelForSequenceClassification.from_pretrained, backbone, num_labels=1
    )
    model.resize_token_embeddings(max(len(tokenizer), model.config.vocab_size))
    return model.to(device)


def address_idf(frames: list[pd.DataFrame]) -> dict[str, float]:
    """IDF table used to summarize long addresses instead of cutting the tail."""
    documents = []
    for frame in frames:
        sample = frame.address_roman
        if len(sample) > 100_000:
            sample = sample.sample(100_000, random_state=config.SEED)
        documents.extend(sample.tolist())
    vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"[^\W_]{2,}", min_df=2)
    vectorizer.fit(documents)
    return {token: float(weight) for token, weight in zip(vectorizer.get_feature_names_out(), vectorizer.idf_)}


def _field(row, key: str):
    """Read a prepared-record field. Series.name is the index label, not the column."""
    if isinstance(row, pd.Series):
        if key not in row.index:
            return ""
        value = row[key]
        return "" if value is None else str(value)
    value = getattr(row, key, "")
    return "" if value is None else str(value)


def record_text(
    row,
    dk_tags: bool,
    include_country: bool,
    idf: dict[str, float] | None,
    drop_address: bool = False,
) -> str:
    address = "" if drop_address else _field(row, "address")
    address_roman = "" if drop_address else _field(row, "address_roman")
    name = _field(row, "name")
    name_roman = _field(row, "name_roman")
    if "name_norm" in getattr(row, "_fields", ()) or (isinstance(row, pd.Series) and "name_norm" in row.index):
        shown_name = _field(row, "name_norm") or name_roman
        if shown_name and shown_name != name_roman:
            name_roman = f"{name_roman} {shown_name}"
    if idf and address_roman:
        address = text.summarize_address(address, idf, config.ADDRESS_SUMMARY_WORDS)
        address_roman = text.summarize_address(address_roman, idf, config.ADDRESS_SUMMARY_WORDS)
    extras = {}
    for key in ("state", "city", "house"):
        try:
            extras[key] = _field(row, key)
        except Exception:
            extras[key] = ""
    return text.serialize(
        name, address, name_roman, address_roman, _field(row, "country"),
        dk_tags=dk_tags, include_country=include_country, extras=extras,
    )


def lookup_frames(split: str) -> dict[str, pd.DataFrame]:
    frames = {f"s{source}": pair_tables.load_prepared(split, source).set_index("id", drop=False) for source in (1, 2, 3)}
    return frames


def _serialize_needed(frame: pd.DataFrame, needed: np.ndarray, dk_tags: bool, include_country: bool, idf) -> dict[str, str]:
    subset = frame.loc[frame.index.isin(needed)]
    return {row.id: record_text(row, dk_tags, include_country, idf) for row in subset.itertuples(index=False)}


def pair_texts(
    table: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    dk_tags: bool = config.DK_TAGS,
    include_country: bool = config.INCLUDE_COUNTRY,
    idf: dict[str, float] | None = None,
    attr_del_p: float = 0.0,
    seed: int = config.SEED,
    augment: bool = False,
) -> tuple[list[str], list[str]]:
    left_map = _serialize_needed(frames["s1"], table.s1_id.unique(), dk_tags, include_country, idf)
    right_map: dict[str, str] = {}
    for source, frame in ((2, frames["s2"]), (3, frames["s3"])):
        needed = table.loc[table.source == source, "cand_id"].unique()
        right_map.update(_serialize_needed(frame, needed, dk_tags, include_country, idf))
    rng = np.random.default_rng(seed)
    drop_p = max(attr_del_p, config.ADDRESS_DROP_P if augment else 0.0)
    labels = table.label.to_numpy() if "label" in table.columns else np.zeros(len(table))
    drop = (rng.random(len(table)) < drop_p) & (labels == 1) if drop_p > 0 else np.zeros(len(table), dtype=bool)
    country_drop = rng.random(len(table)) < config.COUNTRY_DROPOUT_P if augment else np.zeros(len(table), dtype=bool)
    ocr = rng.random(len(table)) < config.OCR_NOISE_P if augment else np.zeros(len(table), dtype=bool)
    left = [left_map[s1_id] for s1_id in table.s1_id.to_numpy()]
    right = []
    for index, (row, drop_address) in enumerate(zip(table.itertuples(index=False), drop)):
        include = include_country and not bool(country_drop[index])
        if drop_address or not include:
            rec = frames[f"s{int(row.source)}"].loc[row.cand_id]
            if isinstance(rec, pd.DataFrame):
                rec = rec.iloc[0]
            text_right = record_text(rec, dk_tags, include, idf, drop_address=bool(drop_address))
            if not include:
                left[index] = record_text(frames["s1"].loc[row.s1_id], dk_tags, False, idf)
        else:
            text_right = right_map[row.cand_id]
        if ocr[index]:
            left[index] = text.ocr_perturb(left[index], rng)
            text_right = text.ocr_perturb(text_right, rng)
        right.append(text_right)
    return left, right


def length_order(left: list[str], right: list[str]) -> np.ndarray:
    return np.argsort([len(a) + len(b) for a, b in zip(left, right)], kind="stable")


def predict_logits(
    model: torch.nn.Module,
    tokenizer,
    left: list[str],
    right: list[str],
    batch_size: int = config.PREDICT_BATCH,
    device: torch.device | str = "cuda",
) -> np.ndarray:
    """Length-sorted bf16 inference. Returns logits in the original pair order."""
    model.eval()
    order = length_order(left, right)
    out = np.zeros(len(left), dtype=np.float32)
    for start in range(0, len(left), batch_size):
        rows = order[start : start + batch_size]
        batch = tokenizer(
            [left[i] for i in rows],
            [right[i] for i in rows],
            max_length=config.MAX_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits.reshape(-1).float()
        out[rows] = logits.cpu().numpy()
        if start == 0 or (start // batch_size) % 20 == 0:
            print(f"score {min(start + batch_size, len(left))}/{len(left)}", flush=True)
    return out



def write_token_cache(left: list[str], right: list[str], labels, tokenizer, folder: Path) -> int:
    """Tokenize pairs once with max-length padding. Rank 0 only."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    n = len(left)
    ids = np.memmap(folder / "input_ids.npy", dtype=np.int32, mode="w+", shape=(n, config.MAX_LEN))
    mask = np.memmap(folder / "attention_mask.npy", dtype=np.int8, mode="w+", shape=(n, config.MAX_LEN))
    for start in range(0, n, config.TOKENIZE_BATCH):
        end = min(start + config.TOKENIZE_BATCH, n)
        encoded = tokenizer(
            left[start:end],
            right[start:end],
            max_length=config.MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        ids[start:end] = encoded["input_ids"].astype(np.int32)
        mask[start:end] = encoded["attention_mask"].astype(np.int8)
        if start == 0 or (start // config.TOKENIZE_BATCH) % 20 == 0:
            print(f"tokenize {end}/{n}", flush=True)
    ids.flush()
    mask.flush()
    np.save(folder / "labels.npy", np.asarray(labels, dtype=np.float32))
    (folder / "done.json").write_text(json.dumps({"rows": n, "max_len": config.MAX_LEN}))
    return n


def load_token_cache(folder: Path) -> dict:
    folder = Path(folder)
    n = int(json.loads((folder / "done.json").read_text())["rows"])
    cache = {
        "input_ids": np.memmap(folder / "input_ids.npy", dtype=np.int32, mode="r", shape=(n, config.MAX_LEN)),
        "attention_mask": np.memmap(folder / "attention_mask.npy", dtype=np.int8, mode="r", shape=(n, config.MAX_LEN)),
        "labels": np.load(folder / "labels.npy"),
    }
    # XLM-R / MiniLM ignore segment ids; omit them so forward() does not reject the kwarg.
    return cache


def _batch_from_cache(cache: dict, rows: np.ndarray, device) -> dict:
    batch = {
        "input_ids": torch.from_numpy(np.asarray(cache["input_ids"][rows])).to(device, dtype=torch.long),
        "attention_mask": torch.from_numpy(np.asarray(cache["attention_mask"][rows])).to(device, dtype=torch.long),
    }
    if "token_type_ids" in cache:
        batch["token_type_ids"] = torch.from_numpy(np.asarray(cache["token_type_ids"][rows])).to(
            device, dtype=torch.long
        )
    return batch


def predict_logits_cache(model: torch.nn.Module, cache: dict, device, batch_size: int = config.PREDICT_BATCH) -> np.ndarray:
    model.eval()
    n = len(cache["labels"])
    lengths = np.asarray(cache["attention_mask"]).sum(axis=1)
    order = np.argsort(lengths, kind="stable")
    out = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        rows = order[start : start + batch_size]
        batch = _batch_from_cache(cache, rows, device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits.reshape(-1).float()
        out[rows] = logits.cpu().numpy()
    return out


def train(
    model: torch.nn.Module,
    tokenizer,
    train_cache: dict,
    calib_cache: dict,
    epochs: int,
    checkpoint_dir: Path,
    device: torch.device | str = "cuda",
    rank: int = 0,
    world: int = 1,
) -> dict:
    """AdamW + linear decay, bf16, keep the best calibration NLL checkpoint.

    BATCH_SIZE is per GPU. Global batch is BATCH_SIZE * world.
    """
    set_seed()
    model = model.to(device)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    model.train()
    n = len(train_cache["labels"])
    lengths = np.asarray(train_cache["attention_mask"]).sum(axis=1)
    order = np.argsort(lengths, kind="stable")
    labels = torch.tensor(train_cache["labels"], dtype=torch.float32)
    micro = config.BATCH_SIZE
    global_batch = config.BATCH_SIZE * world
    steps_per_epoch = max(1, n // global_batch)
    total_steps = steps_per_epoch * epochs
    optimizer = AdamW(distributed.unwrap(model).parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.WARMUP_FRACTION),
        num_training_steps=total_steps,
    )
    best_loss = float("inf")
    history: list[dict] = []
    step = 0
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"train n={n} global_batch={global_batch} steps/epoch={steps_per_epoch} world={world}", flush=True)
    distributed.barrier()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        for start in range(0, steps_per_epoch * global_batch, global_batch):
            global_rows = order[start : start + global_batch]
            rows = global_rows[rank * micro : (rank + 1) * micro]
            batch = _batch_from_cache(train_cache, rows, device)
            target = labels[rows].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits.reshape(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            running += float(loss.item()) * len(rows)
            seen += len(rows)
            step += 1
            if rank == 0 and step % 50 == 0:
                print(f"epoch {epoch + 1} step {step}/{total_steps} loss {running / seen:.4f}", flush=True)
            at_epoch_end = start + global_batch >= steps_per_epoch * global_batch
            if step % config.EVAL_EVERY_STEPS == 0 or at_epoch_end:
                if rank == 0:
                    calib_loss = _calibration_nll_cache(distributed.unwrap(model), calib_cache, device)
                    row = {"epoch": epoch + 1, "step": step, "train_loss": running / max(seen, 1), "calib_nll": calib_loss}
                    history.append(row)
                    print(row, flush=True)
                    if calib_loss < best_loss:
                        best_loss = calib_loss
                        save(distributed.unwrap(model), tokenizer, checkpoint_dir)
                        (checkpoint_dir / "best.json").write_text(json.dumps({**row, "best_calib_nll": best_loss}, indent=2))
                distributed.barrier()
        order = _shuffle_buckets(order, epoch, global_batch)

    return {"best_calib_nll": best_loss, "history": history, "steps": step}


def _shuffle_buckets(order: np.ndarray, epoch: int, global_batch: int) -> np.ndarray:
    """Shuffle length buckets so similar-length pairs stay in the same global step."""
    rng = np.random.default_rng(config.SEED + epoch + 1)
    n = len(order)
    n_full = (n // global_batch) * global_batch
    if n_full == 0:
        return order.copy()
    buckets = order[:n_full].reshape(-1, global_batch)
    rng.shuffle(buckets)
    if n_full == n:
        return buckets.ravel()
    return np.concatenate([buckets.ravel(), order[n_full:]])


def _calibration_nll_cache(model: torch.nn.Module, cache: dict, device) -> float:
    model.eval()
    logits = predict_logits_cache(model, cache, device)
    y = torch.tensor(cache["labels"], dtype=torch.float32)
    z = torch.tensor(logits, dtype=torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    model.train()
    return float(loss.item())


def save(model: torch.nn.Module, tokenizer, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    distributed.unwrap(model).save_pretrained(path)
    tokenizer.save_pretrained(path)


def load_checkpoint(path: Path, device: torch.device | str = "cuda"):
    tokenizer = _from_pretrained(AutoTokenizer.from_pretrained, str(path))
    model = _from_pretrained(AutoModelForSequenceClassification.from_pretrained, str(path))
    return tokenizer, model.to(device)
