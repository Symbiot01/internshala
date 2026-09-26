"""Stage 5 and 6: bake-off, full Ditto training, calibration, decision, robustness.

Writes reports/04_*.json and the matcher checkpoint under models/matcher/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import calibrate
import config
import cross_encoder
import data
import decide
import distributed
import pairs as pair_tables
import text


def write_report(name: str, report: dict) -> None:
    if not distributed.is_main():
        return
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / f"{name}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def load_benchmark() -> dict:
    path = config.REPORTS_DIR / "00_benchmark.json"
    return json.loads(path.read_text()) if path.exists() else {}


def build_idf():
    frames = [pair_tables.load_prepared("train", source) for source in (1, 2, 3)]
    return cross_encoder.address_idf(frames)


def score_table(model, tokenizer, table, frames, idf, include_country: bool, device, rank: int = 0, world: int = 1) -> np.ndarray:
    """Score pair rows; ranks take strided slices and rank 0 concatenates in order."""
    n = len(table)
    mine = table.iloc[np.arange(n) % world == rank].reset_index(drop=True)
    left, right = cross_encoder.pair_texts(mine, frames, include_country=include_country, idf=idf)
    part = cross_encoder.predict_logits(model, tokenizer, left, right, device=device)
    if world == 1:
        return part
    path = config.CACHE_DIR / "train" / f"score_part_{rank}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, part)
    distributed.barrier()
    if rank != 0:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    for r in range(world):
        out[np.arange(n) % world == r] = np.load(config.CACHE_DIR / "train" / f"score_part_{r}.npy")
    return out


def assignment_from_table(table, probs, rule, truth) -> tuple[dict[str, list[str]], dict]:
    packed = decide.pack(table.s1_id.to_numpy(), table.cand_id.to_numpy(), probs, table.label.to_numpy(), truth)
    triples = decide.select(packed, rule["threshold"], rule["cap"], rule["alpha"], rule["gate"])
    assigned = decide.unique_mapping(triples)
    entities = np.unique(table.s1_id.to_numpy())
    summary = decide.evaluate_assignment(assigned, entities, truth)
    return assigned, {**rule, **summary, "umc_pairs": sum(len(v) for v in assigned.values())}


def run_bakeoff(frames, idf, include_country: bool, device, rank: int, world: int) -> str:
    train_pairs = pair_tables.sample_train_entities(pair_tables.load_pairs("train"), config.BAKEOFF_ENTITIES)
    calib_pairs = pair_tables.sample_train_entities(pair_tables.load_pairs("calibration"), 8_000, seed=config.SEED + 1)
    truth = data.read_truth("train")
    bench = load_benchmark()
    test_entities = 1_732_544
    keep = json.loads((config.REPORTS_DIR / "02_fuse_train.json").read_text()).get("keep_per_source", 20)
    results = {}
    for key, repo in config.BACKBONES.items():
        print(f"bake-off {key}", flush=True)
        tokenizer = cross_encoder.load_tokenizer()
        model = cross_encoder.load_model(repo, tokenizer, device=device)
        train_dir = config.CACHE_DIR / "train" / f"token_cache_bake_{key}"
        calib_dir = config.CACHE_DIR / "train" / f"token_cache_bake_calib_{key}"
        if rank == 0:
            left, right = cross_encoder.pair_texts(train_pairs, frames, include_country=include_country, idf=idf)
            cross_encoder.write_token_cache(left, right, train_pairs.label.to_numpy(), tokenizer, train_dir)
            calib_left, calib_right = cross_encoder.pair_texts(calib_pairs, frames, include_country=include_country, idf=idf)
            cross_encoder.write_token_cache(calib_left, calib_right, calib_pairs.label.to_numpy(), tokenizer, calib_dir)
        distributed.wait_for_file(train_dir / "done.json", rank)
        distributed.wait_for_file(calib_dir / "done.json", rank)
        ckpt = config.MODELS_DIR / f"bakeoff_{key}"
        history = cross_encoder.train(
            model, tokenizer, cross_encoder.load_token_cache(train_dir), cross_encoder.load_token_cache(calib_dir),
            epochs=1, checkpoint_dir=ckpt, device=device, rank=rank, world=world,
        )
        tokenizer, model = cross_encoder.load_checkpoint(ckpt, device=device)
        logits = score_table(model, tokenizer, calib_pairs, frames, idf, include_country, device, rank, world)
        if rank == 0:
            temperature = calibrate.fit_temperature(logits, calib_pairs.label.to_numpy())
            probs = calibrate.probabilities(logits, temperature)
            packed = decide.pack(calib_pairs.s1_id.to_numpy(), calib_pairs.cand_id.to_numpy(), probs, calib_pairs.label.to_numpy(), truth)
            rule = decide.grid_search(packed)
            _, summary = assignment_from_table(calib_pairs, probs, rule, truth)
            predict_rate = bench.get(f"{key}_predict_pairs_per_second", 3000)
            projected_hours = (test_entities * 2 * keep / max(predict_rate, 1)) / 3600
            results[key] = {
                **history,
                **summary,
                "temperature": temperature,
                "projected_test_hours": round(projected_hours, 2),
                "predict_pairs_per_second": predict_rate,
            }
        del model
        import torch
        torch.cuda.empty_cache()
        distributed.barrier()
    write_report("04_bakeoff", results)
    if rank == 0:
        viable = {k: v for k, v in results.items() if v["projected_test_hours"] <= 8.0}
        pool = viable or results
        chosen = max(pool, key=lambda k: pool[k]["macro_f05"])
    else:
        chosen = "minilm"
    return distributed.broadcast_object(chosen, rank, world)


def country_flags(entities: np.ndarray, s1) -> np.ndarray:
    lookup = dict(zip(s1.id.tolist(), s1.country.tolist()))
    return np.array([lookup.get(entity, "") for entity in entities])


def leave_one_country(frames, idf, include_country: bool, device, rank: int, world: int) -> dict:
    """Train MiniLM on one country, score the other country's hold-out entities."""
    train_pairs = pair_tables.load_pairs("train")
    holdout = pair_tables.load_pairs("evaluation")
    s1 = pair_tables.load_prepared("train", 1)
    train_country = s1.set_index("id").loc[train_pairs.s1_id].country.to_numpy()
    eval_country = s1.set_index("id").loc[holdout.s1_id].country.to_numpy()
    countries = sorted(set(s1.country.unique().tolist()))
    report = {"countries": countries}
    if len(countries) < 2:
        return report
    truth = data.read_truth("train")
    a, b = countries[0], countries[1]
    for source_country, target_country in ((a, b), (b, a)):
        train_sub = train_pairs[train_country == source_country]
        train_sub = pair_tables.sample_train_entities(train_sub, min(config.BAKEOFF_ENTITIES, train_sub.s1_id.nunique()))
        eval_sub = holdout[eval_country == target_country]
        eval_sub = pair_tables.sample_train_entities(eval_sub, min(8_000, eval_sub.s1_id.nunique()), seed=config.SEED + 3)
        if train_sub.empty or eval_sub.empty:
            continue
        tokenizer = cross_encoder.load_tokenizer()
        model = cross_encoder.load_model(config.BACKBONES["minilm"], tokenizer, device=device)
        train_dir = config.CACHE_DIR / "train" / f"token_cache_loc_{source_country}_{target_country}"
        eval_dir = config.CACHE_DIR / "train" / f"token_cache_loc_eval_{source_country}_{target_country}"
        if rank == 0:
            left, right = cross_encoder.pair_texts(train_sub, frames, include_country=include_country, idf=idf)
            cross_encoder.write_token_cache(left, right, train_sub.label.to_numpy(), tokenizer, train_dir)
            el, er = cross_encoder.pair_texts(eval_sub, frames, include_country=include_country, idf=idf)
            cross_encoder.write_token_cache(el, er, eval_sub.label.to_numpy(), tokenizer, eval_dir)
        distributed.wait_for_file(train_dir / "done.json", rank)
        distributed.wait_for_file(eval_dir / "done.json", rank)
        ckpt = config.MODELS_DIR / f"loc_{source_country}_to_{target_country}"
        cross_encoder.train(
            model, tokenizer, cross_encoder.load_token_cache(train_dir), cross_encoder.load_token_cache(eval_dir),
            epochs=1, checkpoint_dir=ckpt, device=device, rank=rank, world=world,
        )
        tokenizer, model = cross_encoder.load_checkpoint(ckpt, device=device)
        logits = score_table(model, tokenizer, eval_sub, frames, idf, include_country, device, rank, world)
        if rank == 0:
            temperature = calibrate.fit_temperature(logits, eval_sub.label.to_numpy())
            probs = calibrate.probabilities(logits, temperature)
            packed = decide.pack(eval_sub.s1_id.to_numpy(), eval_sub.cand_id.to_numpy(), probs, eval_sub.label.to_numpy(), truth)
            rule = decide.grid_search(packed)
            _, summary = assignment_from_table(eval_sub, probs, rule, truth)
            report[f"{source_country}->{target_country}"] = summary
        del model
        import torch
        torch.cuda.empty_cache()
        distributed.barrier()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bakeoff", action="store_true")
    parser.add_argument("--backbone", choices=list(config.BACKBONES), default=None)
    parser.add_argument("--include-country", action="store_true", default=config.INCLUDE_COUNTRY)
    parser.add_argument("--no-country", action="store_true")
    args = parser.parse_args()
    include_country = False if args.no_country else args.include_country
    distributed.maybe_relaunch()
    rank, world, device = distributed.setup()

    for group in ("train", "calibration", "evaluation"):
        if rank == 0 and not pair_tables.pair_path(group).exists():
            stats = pair_tables.write_pairs(group)
            write_report(f"04_pairs_{group}", stats)
    distributed.barrier()

    frames = cross_encoder.lookup_frames("train")
    idf = build_idf()
    if args.backbone:
        chosen = args.backbone
        write_report("04_bakeoff", {"skipped": True, "chosen": chosen})
    elif args.skip_bakeoff:
        chosen = "minilm"
        write_report("04_bakeoff", {"skipped": True, "chosen": chosen, "reason": "--skip-bakeoff"})
    else:
        chosen = run_bakeoff(frames, idf, include_country, device, rank, world)
    write_report("04_backbone", {"chosen": chosen, "repo": config.BACKBONES[chosen], "include_country": include_country})

    train_pairs = pair_tables.load_pairs("train")
    calib_pairs = pair_tables.load_pairs("calibration")
    eval_pairs = pair_tables.load_pairs("evaluation")
    tokenizer = cross_encoder.load_tokenizer()
    train_dir = config.CACHE_DIR / "train" / "token_cache_train"
    calib_dir = config.CACHE_DIR / "train" / "token_cache_calib_nll"
    if rank == 0:
        if not (train_dir / "done.json").exists():
            left, right = cross_encoder.pair_texts(
                train_pairs, frames, include_country=include_country, idf=idf, attr_del_p=config.ATTR_DEL_P, augment=True
            )
            cross_encoder.write_token_cache(left, right, train_pairs.label.to_numpy(), tokenizer, train_dir)
            del left, right
        if not (calib_dir / "done.json").exists():
            calib_for_nll = pair_tables.sample_train_entities(calib_pairs, 4_000, seed=config.SEED + 5)
            if len(calib_for_nll) > config.CALIB_LOSS_PAIRS:
                calib_for_nll = calib_for_nll.sample(config.CALIB_LOSS_PAIRS, random_state=config.SEED)
            calib_left, calib_right = cross_encoder.pair_texts(
                calib_for_nll, frames, include_country=include_country, idf=idf
            )
            cross_encoder.write_token_cache(
                calib_left, calib_right, calib_for_nll.label.to_numpy(), tokenizer, calib_dir
            )
            del calib_left, calib_right
    distributed.wait_for_file(train_dir / "done.json", rank)
    distributed.wait_for_file(calib_dir / "done.json", rank)
    model = cross_encoder.load_model(config.BACKBONES[chosen], tokenizer, device=device)
    ckpt = config.MODELS_DIR / "matcher"
    history = cross_encoder.train(
        model, tokenizer,
        cross_encoder.load_token_cache(train_dir),
        cross_encoder.load_token_cache(calib_dir),
        epochs=config.EPOCHS, checkpoint_dir=ckpt, device=device, rank=rank, world=world,
    )
    write_report("04_train", {"backbone": chosen, **history})

    tokenizer, model = cross_encoder.load_checkpoint(ckpt, device=device)
    calib_logits = score_table(model, tokenizer, calib_pairs, frames, idf, include_country, device, rank, world)
    eval_logits = score_table(model, tokenizer, eval_pairs, frames, idf, include_country, device, rank, world)
    if rank == 0:
        np.savez(config.CACHE_DIR / "train" / "matcher_logits.npz",
                 calib_s1=calib_pairs.s1_id.to_numpy(), calib_cand=calib_pairs.cand_id.to_numpy(),
                 calib_label=calib_pairs.label.to_numpy(), calib_logit=calib_logits,
                 eval_s1=eval_pairs.s1_id.to_numpy(), eval_cand=eval_pairs.cand_id.to_numpy(),
                 eval_label=eval_pairs.label.to_numpy(), eval_logit=eval_logits)

        temperature = calibrate.fit_temperature(calib_logits, calib_pairs.label.to_numpy())
        calib_prob = calibrate.probabilities(calib_logits, temperature)
        eval_prob = calibrate.probabilities(eval_logits, temperature)
        truth = data.read_truth("train")
        packed = decide.pack(calib_pairs.s1_id.to_numpy(), calib_pairs.cand_id.to_numpy(), calib_prob, calib_pairs.label.to_numpy(), truth)
        rule = decide.grid_search(packed)
        _, calib_summary = assignment_from_table(calib_pairs, calib_prob, rule, truth)
        assigned, _ = assignment_from_table(eval_pairs, eval_prob, rule, truth)
        s1 = pair_tables.load_prepared("train", 1)
        groups = np.load(config.CACHE_DIR / "train" / "entities.npz")
        eval_ids = s1.id.to_numpy()[groups["evaluation"]]
        eval_summary = decide.evaluate_assignment(assigned, eval_ids, truth)
        countries = country_flags(eval_ids, s1)
        gold_ids = {mid for entity in eval_ids for mid in truth.get(entity, [])}
        other_names = {}
        for source in (2, 3):
            frame = pair_tables.load_prepared("train", source)
            needed = frame[frame.id.isin(gold_ids)]
            other_names.update({row.id: text.has_non_latin_letters(row.name) for row in needed.itertuples(index=False)})
        sliced = decide.breakdown(assigned, eval_ids, countries, truth, other_names)
    else:
        calib_summary = eval_summary = sliced = {}
        temperature = 1.0
        rule = {}
        ship = False
    loc = leave_one_country(frames, idf, include_country, device, rank, world)
    if rank == 0:
        ship = eval_summary["macro_f05"] >= config.SHIP_BASELINE + config.SHIP_MARGIN
        write_report("04_decision", {
            "temperature": temperature,
            "rule": rule,
            "calibration": calib_summary,
            "evaluation": eval_summary,
            "breakdown": sliced,
            "leave_one_country": loc,
            "ship": ship,
            "ship_rule": f"replace A if eval F0.5 >= {config.SHIP_BASELINE + config.SHIP_MARGIN:.3f}",
        })
    distributed.cleanup()


if __name__ == "__main__":
    main()
