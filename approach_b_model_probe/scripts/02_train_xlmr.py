"""One-epoch Ditto XLM-R on 30k train-split entities. Writes models/xlmr/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import cross_encoder
import distributed
import frames_util
import pairs as pair_tables


def main() -> None:
    ckpt = config.MODELS_DIR / "xlmr"
    if (ckpt / "best.json").exists() or (ckpt / "config.json").exists():
        print(json.dumps({"skipped": True, "reason": "checkpoint exists", "path": str(ckpt)}), flush=True)
        return
    distributed.maybe_relaunch()
    rank, world, device = distributed.setup()
    frames = cross_encoder.lookup_frames("train")
    frames = {k: frames_util.strip_prepared(v) for k, v in frames.items()}
    idf = cross_encoder.address_idf(
        [frames_util.strip_prepared(pair_tables.load_prepared("train", source)) for source in (1, 2, 3)]
    )
    train_pairs = pair_tables.sample_train_entities(pair_tables.load_pairs("train"), config.BAKEOFF_ENTITIES)
    calib_pairs = pair_tables.sample_train_entities(
        pair_tables.load_pairs("calibration"), 8_000, seed=config.SEED + 1
    )
    if len(calib_pairs) > config.CALIB_LOSS_PAIRS:
        calib_pairs = calib_pairs.sample(config.CALIB_LOSS_PAIRS, random_state=config.SEED)
    tok_dir = config.CACHE_DIR / "xlmr_tokenizer"
    if rank == 0:
        tokenizer = cross_encoder.load_tokenizer()
        tokenizer.save_pretrained(tok_dir)
    distributed.wait_for_file(tok_dir / "tokenizer_config.json", rank)
    tokenizer = cross_encoder._from_pretrained(cross_encoder.AutoTokenizer.from_pretrained, str(tok_dir))
    train_dir = config.CACHE_DIR / "token_cache_xlmr_train"
    calib_dir = config.CACHE_DIR / "token_cache_xlmr_calib"
    if rank == 0:
        if not (train_dir / "done.json").exists():
            left, right = cross_encoder.pair_texts(
                train_pairs, frames, include_country=config.INCLUDE_COUNTRY, idf=idf, attr_del_p=0.0, augment=False
            )
            cross_encoder.write_token_cache(left, right, train_pairs.label.to_numpy(), tokenizer, train_dir)
        if not (calib_dir / "done.json").exists():
            left, right = cross_encoder.pair_texts(
                calib_pairs, frames, include_country=config.INCLUDE_COUNTRY, idf=idf, attr_del_p=0.0, augment=False
            )
            cross_encoder.write_token_cache(left, right, calib_pairs.label.to_numpy(), tokenizer, calib_dir)
    distributed.wait_for_file(train_dir / "done.json", rank)
    distributed.wait_for_file(calib_dir / "done.json", rank)
    model = cross_encoder.load_model(config.BACKBONES["xlmr"], tokenizer, device=device)
    history = cross_encoder.train(
        model,
        tokenizer,
        cross_encoder.load_token_cache(train_dir),
        cross_encoder.load_token_cache(calib_dir),
        epochs=config.EPOCHS,
        checkpoint_dir=ckpt,
        device=device,
        rank=rank,
        world=world,
    )
    if rank == 0:
        report = {"backbone": "xlmr", "repo": config.BACKBONES["xlmr"], **history}
        (config.REPORTS_DIR / "02_train_xlmr.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    distributed.cleanup()


if __name__ == "__main__":
    main()
