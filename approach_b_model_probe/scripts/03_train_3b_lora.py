"""Gated LoRA train of Qwen2.5-3B. Skips unless reports/delta.json promote is true."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import cross_encoder
import distributed
import frames_util
import pairs as pair_tables


def _gated() -> bool:
    path = config.REPORTS_DIR / "delta.json"
    if not path.exists():
        return False
    return bool(json.loads(path.read_text()).get("promote"))


def _load_base(name: str, tokenizer):
    model = cross_encoder._from_pretrained(
        AutoModelForSequenceClassification.from_pretrained, name, num_labels=1
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    try:
        model.resize_token_embeddings(max(len(tokenizer), model.config.vocab_size))
    except Exception:
        pass
    return model


def main() -> None:
    ckpt = config.MODELS_DIR / "qwen3b_lora"
    if not _gated():
        report = {"skipped": True, "reason": "XLM-R promote gate failed or delta.json missing"}
        (config.REPORTS_DIR / "03_train_3b.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return
    if (ckpt / "adapter_config.json").exists():
        print(json.dumps({"skipped": True, "reason": "checkpoint exists"}), flush=True)
        return
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise SystemExit("peft is required for Stage B; install it in a probe venv, not approach_b/.venv") from exc

    distributed.maybe_relaunch()
    rank, world, device = distributed.setup()
    frames = cross_encoder.lookup_frames("train")
    frames = {k: frames_util.strip_prepared(v) for k, v in frames.items()}
    idf = cross_encoder.address_idf(
        [frames_util.strip_prepared(pair_tables.load_prepared("train", source)) for source in (1, 2, 3)]
    )
    train_pairs = pair_tables.sample_train_entities(
        pair_tables.load_pairs("train"), config.QWEN_TRAIN_ENTITIES
    )
    if len(train_pairs) > config.QWEN_PAIR_CAP:
        train_pairs = train_pairs.sample(config.QWEN_PAIR_CAP, random_state=config.SEED).reset_index(drop=True)
    calib_pairs = pair_tables.sample_train_entities(
        pair_tables.load_pairs("calibration"), 2_000, seed=config.SEED + 1
    )
    if len(calib_pairs) > 8192:
        calib_pairs = calib_pairs.sample(8192, random_state=config.SEED)

    base_name = config.BACKBONES["qwen3b"]
    try:
        tokenizer = cross_encoder._from_pretrained(AutoTokenizer.from_pretrained, base_name)
        tokenizer.add_tokens(config.SPECIAL_TOKENS, special_tokens=True)
        model = _load_base(base_name, tokenizer)
    except (OSError, ValueError):
        base_name = config.BACKBONES["phi3"]
        tokenizer = cross_encoder._from_pretrained(AutoTokenizer.from_pretrained, base_name, trust_remote_code=True)
        tokenizer.add_tokens(config.SPECIAL_TOKENS, special_tokens=True)
        model = _load_base(base_name, tokenizer)

    lora = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    train_dir = config.CACHE_DIR / "token_cache_qwen_train"
    calib_dir = config.CACHE_DIR / "token_cache_qwen_calib"
    if rank == 0:
        if not (train_dir / "done.json").exists():
            left, right = cross_encoder.pair_texts(train_pairs, frames, idf=idf, augment=False)
            cross_encoder.write_token_cache(left, right, train_pairs.label.to_numpy(), tokenizer, train_dir)
        if not (calib_dir / "done.json").exists():
            left, right = cross_encoder.pair_texts(calib_pairs, frames, idf=idf, augment=False)
            cross_encoder.write_token_cache(left, right, calib_pairs.label.to_numpy(), tokenizer, calib_dir)
    distributed.wait_for_file(train_dir / "done.json", rank)
    distributed.wait_for_file(calib_dir / "done.json", rank)

    # Reuse Ditto trainer: temporarily shrink per-GPU batch for 3B.
    old_batch = config.BATCH_SIZE
    config.BATCH_SIZE = config.QWEN_MICRO_BATCH
    history = cross_encoder.train(
        model,
        tokenizer,
        cross_encoder.load_token_cache(train_dir),
        cross_encoder.load_token_cache(calib_dir),
        epochs=1,
        checkpoint_dir=ckpt,
        device=device,
        rank=rank,
        world=world,
    )
    config.BATCH_SIZE = old_batch
    if rank == 0:
        unwrapped = distributed.unwrap(model)
        unwrapped.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        (ckpt / "probe_meta.json").write_text(json.dumps({"base": base_name, "lora_r": config.LORA_R}))
        (config.REPORTS_DIR / "03_train_3b.json").write_text(
            json.dumps({"base": base_name, **history}, indent=2)
        )
        print(json.dumps({"base": base_name, **history}, indent=2), flush=True)
    distributed.cleanup()


if __name__ == "__main__":
    main()
