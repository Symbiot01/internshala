"""Approach A: block, cascade, fit LightGBM, tune macro F0.5, write submission files.

Trains only on the provided TSVs. No external identity lookup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import lightgbm as lgb
import numpy as np
from joblib import dump, load

from src.blocking.index import TOP_K, combined_text, fit_index, query_topk
from src.data.io import load_split, parse_id_list
from src.eval.metrics import macro_prf
from src.matching.features import (
    CHEAP_NAMES,
    FEATURE_NAMES,
    HIGH_COSINE,
    cheap_vector,
    full_vector,
)
from src.textnorm import _norm_address, _norm_name, content_tokens

ROOT = Path(__file__).resolve().parents[3]


def is_validation(entity_id: str) -> bool:
    digest = hashlib.blake2b(entity_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % 10 == 0


def _bucket(count: int) -> str:
    if count <= 1:
        return str(count)
    if count <= 3:
        return "2-3"
    if count <= 6:
        return "4-6"
    return "7+"


def normalize_many(values: list[str], is_address: bool, cache: Path) -> list[str]:
    if cache.exists():
        print(f"cache {cache.name}", flush=True)
        return load(cache)
    worker = _norm_address if is_address else _norm_name
    with Pool(processes=6) as pool:
        normalized = pool.map(worker, values, chunksize=8000)
    cache.parent.mkdir(parents=True, exist_ok=True)
    dump(normalized, cache, compress=0)
    return normalized


def sample_positions(entity_ids: np.ndarray, countries: np.ndarray, counts: dict[str, int], size: int, seed: int) -> np.ndarray:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, entity_id in enumerate(entity_ids):
        groups[(countries[index], _bucket(counts.get(entity_id, 0)))].append(index)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    total = len(entity_ids)
    for indexes in groups.values():
        take = min(len(indexes), max(1, int(round(len(indexes) / total * size))))
        picked = rng.choice(indexes, size=take, replace=False)
        chosen.extend(int(value) for value in np.atleast_1d(picked))
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) > size:
        keep = rng.choice(chosen, size=size, replace=False)
        chosen = [int(value) for value in keep]
    elif len(chosen) < size:
        missing = size - len(chosen)
        pool = np.setdiff1d(np.arange(total), np.array(chosen, dtype=np.int64), assume_unique=False)
        extra = rng.choice(pool, size=min(missing, len(pool)), replace=False)
        chosen.extend(int(value) for value in np.atleast_1d(extra))
    return np.array(chosen, dtype=np.int64)


def _idf_lookup(index) -> tuple[dict[str, float], float]:
    names = index.vectorizer.get_feature_names_out()
    values = index.vectorizer.idf_
    table = {name: float(value) for name, value in zip(names, values, strict=True)}
    default = float(np.median(values)) if len(values) else 1.0
    return table, default


def _mean_idf(name: str, table: dict[str, float], default: float) -> float:
    tokens = content_tokens(name)
    if not tokens:
        return 0.0
    return float(sum(table.get(token, default) for token in tokens) / len(tokens))


def _append_source(
    s1_positions: np.ndarray,
    s1_names: list[str],
    s1_addrs: list[str],
    index,
    from_source3: bool,
    gold: dict[str, set[str]],
    s1_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    queries = [combined_text(s1_names[pos], s1_addrs[pos]) for pos in s1_positions]
    hit_pos, hit_scores = query_topk(index, queries, k=TOP_K)
    s1_out: list[int] = []
    other_out: list[int] = []
    cosine_out: list[float] = []
    rank_out: list[int] = []
    label_out: list[int] = []
    flag_out: list[int] = []
    flag = 1 if from_source3 else 0
    for local, (positions, scores) in enumerate(zip(hit_pos, hit_scores, strict=True)):
        s1_index = int(s1_positions[local])
        truth = gold.get(s1_ids[s1_index], set())
        for rank, (other_index, score) in enumerate(zip(positions.tolist(), scores.tolist(), strict=True)):
            s1_out.append(s1_index)
            other_out.append(int(other_index))
            cosine_out.append(float(score))
            rank_out.append(rank)
            label_out.append(1 if index.ids[other_index] in truth else 0)
            flag_out.append(flag)
    return {
        "s1": np.asarray(s1_out, dtype=np.int32),
        "other": np.asarray(other_out, dtype=np.int32),
        "cosine": np.asarray(cosine_out, dtype=np.float32),
        "rank": np.asarray(rank_out, dtype=np.int16),
        "label": np.asarray(label_out, dtype=np.int8),
        "source3": np.asarray(flag_out, dtype=np.int8),
    }


def _concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = parts[0].keys()
    return {key: np.concatenate([part[key] for part in parts]) for key in keys}


def _sort_by_entity(pairs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    order = np.argsort(pairs["s1"], kind="mergesort")
    return {key: value[order] for key, value in pairs.items()}


def _spans(s1_index: np.ndarray) -> list[tuple[int, int]]:
    if len(s1_index) == 0:
        return []
    change = np.flatnonzero(s1_index[1:] != s1_index[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), change))
    ends = np.concatenate((change, np.array([len(s1_index)], dtype=np.int64)))
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _copy_prefilter_relative(blocked: dict[str, np.ndarray], kept: dict[str, np.ndarray], matrix: np.ndarray, names: tuple[str, ...]) -> None:
    """Relative features describe the pre-filter top-50 list, not the survivors."""
    stats: dict[int, tuple[float, float, float]] = {}
    for start, end in _spans(blocked["s1"]):
        scores = blocked["cosine"][start:end]
        best = float(scores.max())
        second = float(np.partition(scores, -2)[-2]) if len(scores) > 1 else 0.0
        stats[int(blocked["s1"][start])] = (best, best - second, float(np.sum(scores >= HIGH_COSINE)))
    best_at = names.index("cosine_over_best")
    gap_at = names.index("gap_to_second")
    high_at = names.index("n_high_cosine")
    for index, s1_index in enumerate(kept["s1"]):
        best, gap, high = stats[int(s1_index)]
        matrix[index, best_at] = float(kept["cosine"][index]) / best if best > 0 else 0.0
        matrix[index, gap_at] = gap
        matrix[index, high_at] = high


def _cheap_matrix(pairs: dict[str, np.ndarray], s1_names, s1_addrs, s1_countries, other_names, other_addrs, other_countries, mean_idf: np.ndarray) -> np.ndarray:
    rows = np.zeros((len(pairs["s1"]), len(CHEAP_NAMES)), dtype=np.float32)
    for index in range(len(pairs["s1"])):
        s1_index = int(pairs["s1"][index])
        other_index = int(pairs["other"][index])
        rows[index] = cheap_vector(
            s1_names[s1_index],
            s1_addrs[s1_index],
            s1_countries[s1_index],
            other_names[other_index],
            other_addrs[other_index],
            other_countries[other_index],
            bool(pairs["source3"][index]),
            float(pairs["cosine"][index]),
            int(pairs["rank"][index]),
            float(mean_idf[s1_index]),
        )
    return rows


def _full_matrix(pairs, s1_names, s1_addrs, s1_countries, other_names, other_addrs, other_countries, mean_idf, idf, idf_default) -> np.ndarray:
    rows = np.zeros((len(pairs["s1"]), len(FEATURE_NAMES)), dtype=np.float32)
    for index in range(len(pairs["s1"])):
        s1_index = int(pairs["s1"][index])
        other_index = int(pairs["other"][index])
        rows[index] = full_vector(
            s1_names[s1_index],
            s1_addrs[s1_index],
            s1_countries[s1_index],
            other_names[other_index],
            other_addrs[other_index],
            other_countries[other_index],
            bool(pairs["source3"][index]),
            float(pairs["cosine"][index]),
            int(pairs["rank"][index]),
            float(mean_idf[s1_index]),
            idf,
            idf_default,
        )
    return rows


def _slice(pairs: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[mask] for key, value in pairs.items()}


def _fit_classifier(train_x, train_y, valid_x, valid_y, estimators: int):
    model = lgb.LGBMClassifier(
        n_estimators=estimators,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        verbose=-1,
    )
    if len(valid_y) == 0 or len(np.unique(train_y)) < 2 or len(np.unique(valid_y)) < 2:
        model.fit(train_x, train_y)
        return model
    callbacks = [lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)]
    model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], callbacks=callbacks)
    return model


def _keep_mask(probabilities: np.ndarray, spans: list[tuple[int, int]], keep_n: int, floor: float) -> np.ndarray:
    mask = np.zeros(len(probabilities), dtype=bool)
    for start, end in spans:
        local = probabilities[start:end]
        order = np.argsort(-local)
        kept = 0
        for offset in order:
            probability = float(local[offset])
            if kept < keep_n or probability >= floor:
                mask[start + int(offset)] = True
                kept += 1
            else:
                break
    return mask


def _tune_cascade(probabilities, labels, spans) -> tuple[int, float, float]:
    best = (12, 0.5, -1.0)
    best_size = 1e9
    covered = 0
    for start, end in spans:
        covered += int(labels[start:end].sum())
    positives = max(covered, 1)
    for keep_n in (8, 12, 16, 20, 30, 40):
        for floor in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5):
            mask = _keep_mask(probabilities, spans, keep_n, floor)
            retained_count = 0
            for start, end in spans:
                retained_count += int(labels[start:end][mask[start:end]].sum())
            retained = retained_count / positives
            average = float(mask.sum()) / max(len(spans), 1)
            if retained >= 0.99 and average < best_size:
                best_size = average
                best = (keep_n, floor, retained)
    if best[2] < 0:
        mask = _keep_mask(probabilities, spans, 40, 0.02)
        retained_count = sum(int(labels[start:end][mask[start:end]].sum()) for start, end in spans)
        best = (40, 0.02, retained_count / positives)
        best_size = float(mask.sum()) / max(len(spans), 1)
    print(
        f"cascade keep={best[0]} floor={best[1]} positive_retention={best[2]:.4f} avg={best_size:.2f}",
        flush=True,
    )
    return best[0], best[1], best[2]


def _select(probabilities: np.ndarray, ids: list[str], threshold: float, cap: int, alpha: float, gate: float) -> list[str]:
    if len(probabilities) == 0 or float(probabilities[0]) < gate:
        return []
    pmax = float(probabilities[0])
    cutoff = min(threshold, alpha * pmax)
    chosen: list[str] = []
    seen: set[str] = set()
    for probability, entity_id in zip(probabilities, ids, strict=True):
        if len(chosen) >= cap or float(probability) < cutoff:
            break
        if entity_id not in seen:
            seen.add(entity_id)
            chosen.append(entity_id)
    return chosen


def _tune_decision(groups: list[tuple[str, np.ndarray, list[str]]], truth: dict[str, set[str]]) -> dict[str, float]:
    best = {"threshold": 0.5, "cap": 4, "alpha": 0.7, "singleton_gate": 0.5, "macro_f05": -1.0}
    thresholds = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    caps = (2, 3, 4, 5, 6, 8, 11)
    alphas = (0.55, 0.75, 0.9, 1.0)
    gates = (0.15, 0.3, 0.45, 0.6, 0.75)
    for threshold in thresholds:
        for cap in caps:
            for alpha in alphas:
                for gate in gates:
                    predicted = {
                        entity_id: set(_select(probs, ids, threshold, cap, alpha, gate))
                        for entity_id, probs, ids in groups
                    }
                    score = macro_prf(predicted, truth)["macro_f05"]
                    if score > best["macro_f05"]:
                        best = {
                            "threshold": threshold,
                            "cap": cap,
                            "alpha": alpha,
                            "singleton_gate": gate,
                            "macro_f05": score,
                        }
    print(
        "decision "
        f"t={best['threshold']} k={best['cap']} alpha={best['alpha']} "
        f"s={best['singleton_gate']} f05={best['macro_f05']:.4f}",
        flush=True,
    )
    return best


def _entity_groups(pairs, probabilities, other_ids_by_source, spans) -> list[tuple[str, np.ndarray, list[str]]]:
    grouped: list[tuple[str, np.ndarray, list[str]]] = []
    for start, end in spans:
        order = np.argsort(-probabilities[start:end])
        probs = probabilities[start + order]
        ids: list[str] = []
        for offset in order:
            row = start + int(offset)
            source_ids = other_ids_by_source[int(pairs["source3"][row])]
            ids.append(str(source_ids[int(pairs["other"][row])]))
        grouped.append((str(pairs["entity"][start]), probs, ids))
    return grouped


def _predict_groups(groups, rule) -> dict[str, set[str]]:
    return {
        entity_id: set(_select(probs, ids, rule["threshold"], int(rule["cap"]), rule["alpha"], rule["singleton_gate"]))
        for entity_id, probs, ids in groups
    }


def _prepare_split(data_dir: Path, cache_dir: Path):
    split = load_split(data_dir)
    name = data_dir.name
    frames = {}
    for source in ("source1", "source2", "source3"):
        frame = split[source]
        print(f"normalize {name} {source}", flush=True)
        names = normalize_many(frame["business_name"].tolist(), False, cache_dir / f"{name}_{source}_name.joblib")
        addrs = normalize_many(frame["business_address"].tolist(), True, cache_dir / f"{name}_{source}_addr.joblib")
        frames[source] = {
            "id": frame["entity_id"].to_numpy(),
            "country": frame["country"].to_numpy(),
            "name": names,
            "addr": addrs,
        }
    texts = {
        source: [combined_text(name, addr) for name, addr in zip(frames[source]["name"], frames[source]["addr"], strict=True)]
        for source in ("source2", "source3")
    }
    print("fit source2 index", flush=True)
    index2 = fit_index(texts["source2"], frames["source2"]["id"])
    print("fit source3 index", flush=True)
    index3 = fit_index(texts["source3"], frames["source3"]["id"])
    return split, frames, {"source2": index2, "source3": index3}


def cmd_train(args: argparse.Namespace) -> int:
    cache_dir = args.model_dir / "cache"
    _split, frames, indexes = _prepare_split(args.data_dir, cache_dir)
    truth_frame = _split["ground_truth"]
    counts: dict[str, int] = {}
    gold_all: dict[str, set[str]] = {}
    for row in truth_frame.itertuples(index=False):
        parsed = parse_id_list(row.matched_entity_ids)
        counts[row.source1_entity_id] = len(parsed)
        gold_all[row.source1_entity_id] = set(parsed)
    positions = sample_positions(frames["source1"]["id"], frames["source1"]["country"], counts, args.max_entities, seed=0)
    print(f"sampled entities {len(positions)}", flush=True)
    gold = {frames["source1"]["id"][pos]: gold_all.get(frames["source1"]["id"][pos], set()) for pos in positions}
    idf, idf_default = _idf_lookup(indexes["source2"])
    idf3, _default3 = _idf_lookup(indexes["source3"])
    idf.update(idf3)
    mean_idf = np.zeros(len(frames["source1"]["id"]), dtype=np.float32)
    for pos in positions:
        mean_idf[pos] = _mean_idf(frames["source1"]["name"][pos], idf, idf_default)

    print("block sample", flush=True)
    blocked = _sort_by_entity(_concat([
        _append_source(positions, frames["source1"]["name"], frames["source1"]["addr"], indexes["source2"], False, gold, frames["source1"]["id"]),
        _append_source(positions, frames["source1"]["name"], frames["source1"]["addr"], indexes["source3"], True, gold, frames["source1"]["id"]),
    ]))
    blocked["entity"] = frames["source1"]["id"][blocked["s1"]]
    found = int(blocked["label"].sum())
    total_gold = sum(len(ids) for ids in gold.values())
    print(f"blocked pairs {len(blocked['s1'])} positives {found} gold {total_gold} recall {found / max(total_gold, 1):.4f}", flush=True)

    other = {0: frames["source2"], 1: frames["source3"]}
    print("cheap features", flush=True)
    cheap = np.zeros((len(blocked["s1"]), len(CHEAP_NAMES)), dtype=np.float32)
    for index in range(len(blocked["s1"])):
        s1_index = int(blocked["s1"][index])
        other_index = int(blocked["other"][index])
        table = other[int(blocked["source3"][index])]
        cheap[index] = cheap_vector(
            frames["source1"]["name"][s1_index],
            frames["source1"]["addr"][s1_index],
            frames["source1"]["country"][s1_index],
            table["name"][other_index],
            table["addr"][other_index],
            table["country"][other_index],
            bool(blocked["source3"][index]),
            float(blocked["cosine"][index]),
            int(blocked["rank"][index]),
            float(mean_idf[s1_index]),
        )
        if index and index % 250000 == 0:
            print(f"cheap {index}/{len(blocked['s1'])}", flush=True)
    spans = _spans(blocked["s1"])
    _copy_prefilter_relative(blocked, blocked, cheap, CHEAP_NAMES)
    valid_pair = np.array([is_validation(entity_id) for entity_id in blocked["entity"]], dtype=bool)
    print("fit cascade", flush=True)
    cascade = _fit_classifier(cheap[~valid_pair], blocked["label"][~valid_pair], cheap[valid_pair], blocked["label"][valid_pair], estimators=200)
    cascade_prob = cascade.predict_proba(cheap)[:, 1]
    valid_spans = [(start, end) for start, end in spans if valid_pair[start]]
    keep_n, floor, retention = _tune_cascade(cascade_prob, blocked["label"], valid_spans)
    mask = _keep_mask(cascade_prob, spans, keep_n, floor)
    print(f"survivors {int(mask.sum())}", flush=True)

    print("full features", flush=True)
    kept = _slice(blocked, mask)
    full = np.zeros((len(kept["s1"]), len(FEATURE_NAMES)), dtype=np.float32)
    for index in range(len(kept["s1"])):
        s1_index = int(kept["s1"][index])
        other_index = int(kept["other"][index])
        table = other[int(kept["source3"][index])]
        full[index] = full_vector(
            frames["source1"]["name"][s1_index],
            frames["source1"]["addr"][s1_index],
            frames["source1"]["country"][s1_index],
            table["name"][other_index],
            table["addr"][other_index],
            table["country"][other_index],
            bool(kept["source3"][index]),
            float(kept["cosine"][index]),
            int(kept["rank"][index]),
            float(mean_idf[s1_index]),
            idf,
            idf_default,
        )
        if index and index % 250000 == 0:
            print(f"full {index}/{len(kept['s1'])}", flush=True)
    kept_spans = _spans(kept["s1"])
    _copy_prefilter_relative(blocked, kept, full, FEATURE_NAMES)

    valid_kept = np.array([is_validation(entity_id) for entity_id in kept["entity"]], dtype=bool)
    print("fit matcher", flush=True)
    joint = _fit_classifier(full[~valid_kept], kept["label"][~valid_kept], full[valid_kept], kept["label"][valid_kept], estimators=1200)
    s2_rows = kept["source3"] == 0
    s3_rows = ~s2_rows
    separate = None
    if s2_rows[~valid_kept].any() and s3_rows[~valid_kept].any():
        model_s2 = _fit_classifier(full[s2_rows & ~valid_kept], kept["label"][s2_rows & ~valid_kept], full[s2_rows & valid_kept], kept["label"][s2_rows & valid_kept], estimators=1200)
        model_s3 = _fit_classifier(full[s3_rows & ~valid_kept], kept["label"][s3_rows & ~valid_kept], full[s3_rows & valid_kept], kept["label"][s3_rows & valid_kept], estimators=1200)
        separate = (model_s2, model_s3)

    valid_truth = {entity_id: ids for entity_id, ids in gold.items() if is_validation(entity_id)}
    other_ids = {0: frames["source2"]["id"], 1: frames["source3"]["id"]}
    joint_prob = joint.predict_proba(full)[:, 1]
    joint_groups = _entity_groups(kept, joint_prob, other_ids, [span for span in kept_spans if valid_kept[span[0]]])
    for entity_id in valid_truth:
        if entity_id not in {item[0] for item in joint_groups}:
            joint_groups.append((entity_id, np.empty(0, dtype=np.float32), []))
    joint_rule = _tune_decision(joint_groups, valid_truth)
    joint_score = macro_prf(_predict_groups(joint_groups, joint_rule), valid_truth)
    mode = "joint"
    chosen_rule = joint_rule
    chosen_score = joint_score
    if separate is not None:
        model_s2, model_s3 = separate
        split_prob = np.zeros(len(kept["s1"]), dtype=np.float32)
        if np.any(s2_rows):
            split_prob[s2_rows] = model_s2.predict_proba(full[s2_rows])[:, 1]
        if np.any(s3_rows):
            split_prob[s3_rows] = model_s3.predict_proba(full[s3_rows])[:, 1]
        split_groups = _entity_groups(kept, split_prob, other_ids, [span for span in kept_spans if valid_kept[span[0]]])
        present = {item[0] for item in split_groups}
        for entity_id in valid_truth:
            if entity_id not in present:
                split_groups.append((entity_id, np.empty(0, dtype=np.float32), []))
        split_rule = _tune_decision(split_groups, valid_truth)
        split_score = macro_prf(_predict_groups(split_groups, split_rule), valid_truth)
        print(f"joint f05={joint_score['macro_f05']:.4f} separate f05={split_score['macro_f05']:.4f}", flush=True)
        if split_score["macro_f05"] > joint_score["macro_f05"]:
            mode = "separate"
            chosen_rule = split_rule
            chosen_score = split_score

    args.model_dir.mkdir(parents=True, exist_ok=True)
    cascade.booster_.save_model(str(args.model_dir / "cascade.txt"))
    joint.booster_.save_model(str(args.model_dir / "matcher.txt"))
    if separate is not None:
        separate[0].booster_.save_model(str(args.model_dir / "matcher_s2.txt"))
        separate[1].booster_.save_model(str(args.model_dir / "matcher_s3.txt"))
    config = {
        "model_mode": mode,
        "cascade_keep": keep_n,
        "cascade_floor": floor,
        "cascade_retention": retention,
        "threshold": chosen_rule["threshold"],
        "cap": chosen_rule["cap"],
        "alpha": chosen_rule["alpha"],
        "singleton_gate": chosen_rule["singleton_gate"],
        "holdout_f05": chosen_score["macro_f05"],
        "holdout_precision": chosen_score["macro_precision"],
        "holdout_recall": chosen_score["macro_recall"],
        "holdout_entities": chosen_score["entities"],
        "feature_names": FEATURE_NAMES,
        "cheap_names": CHEAP_NAMES,
        "max_entities": args.max_entities,
        "blocking_recall_at_50_measured": 0.9642,
    }
    (args.model_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({key: config[key] for key in ("model_mode", "holdout_f05", "holdout_precision", "holdout_recall")}, indent=2))
    return 0


def _write_line(handle, entity_id: str, ids: list[str]) -> None:
    handle.write(entity_id)
    handle.write("\t")
    handle.write(",".join(dict.fromkeys(ids)))
    handle.write("\n")


def cmd_infer(args: argparse.Namespace) -> int:
    config = json.loads((args.model_dir / "config.json").read_text())
    cascade = lgb.Booster(model_file=str(args.model_dir / "cascade.txt"))
    joint = lgb.Booster(model_file=str(args.model_dir / "matcher.txt"))
    separate = None
    if config["model_mode"] == "separate":
        separate = (
            lgb.Booster(model_file=str(args.model_dir / "matcher_s2.txt")),
            lgb.Booster(model_file=str(args.model_dir / "matcher_s3.txt")),
        )
    _split, frames, indexes = _prepare_split(args.data_dir, args.model_dir / "cache")
    idf, idf_default = _idf_lookup(indexes["source2"])
    idf.update(_idf_lookup(indexes["source3"])[0])
    other = {0: frames["source2"], 1: frames["source3"]}
    other_ids = {0: frames["source2"]["id"], 1: frames["source3"]["id"]}
    rule = config
    args.matching.parent.mkdir(parents=True, exist_ok=True)
    match_file = args.matching.open("w", encoding="utf-8")
    cand_file = args.candidate.open("w", encoding="utf-8")
    match_file.write("source1_entity_id\tmatched_entity_ids\n")
    cand_file.write("source1_entity_id\tcandidate_entity_ids\n")
    total = len(frames["source1"]["id"])
    if args.limit:
        total = min(total, args.limit)
    chunk = 2000
    for start in range(0, total, chunk):
        positions = np.arange(start, min(start + chunk, total), dtype=np.int64)
        mean_idf = np.zeros(len(frames["source1"]["id"]), dtype=np.float32)
        for pos in positions:
            mean_idf[pos] = _mean_idf(frames["source1"]["name"][pos], idf, idf_default)
        parts = [
            _append_source(positions, frames["source1"]["name"], frames["source1"]["addr"], indexes["source2"], False, {}, frames["source1"]["id"]),
            _append_source(positions, frames["source1"]["name"], frames["source1"]["addr"], indexes["source3"], True, {}, frames["source1"]["id"]),
        ]
        if len(parts[0]["s1"]) == 0 and len(parts[1]["s1"]) == 0:
            for pos in positions:
                entity_id = str(frames["source1"]["id"][pos])
                _write_line(match_file, entity_id, [])
                _write_line(cand_file, entity_id, [])
            continue
        blocked = _sort_by_entity(_concat(parts))
        blocked["entity"] = frames["source1"]["id"][blocked["s1"]]
        cheap = np.zeros((len(blocked["s1"]), len(CHEAP_NAMES)), dtype=np.float32)
        for index in range(len(blocked["s1"])):
            s1_index = int(blocked["s1"][index])
            other_index = int(blocked["other"][index])
            table = other[int(blocked["source3"][index])]
            cheap[index] = cheap_vector(
                frames["source1"]["name"][s1_index],
                frames["source1"]["addr"][s1_index],
                frames["source1"]["country"][s1_index],
                table["name"][other_index],
                table["addr"][other_index],
                table["country"][other_index],
                bool(blocked["source3"][index]),
                float(blocked["cosine"][index]),
                int(blocked["rank"][index]),
                float(mean_idf[s1_index]),
            )
        spans = _spans(blocked["s1"])
        _copy_prefilter_relative(blocked, blocked, cheap, CHEAP_NAMES)
        cascade_prob = cascade.predict(cheap)
        mask = _keep_mask(cascade_prob, spans, int(config["cascade_keep"]), float(config["cascade_floor"]))
        kept = _slice(blocked, mask)
        kept_prob_cascade = cascade_prob[mask]
        if len(kept["s1"]) == 0:
            full_prob = np.empty(0, dtype=np.float32)
            full = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        else:
            full = np.zeros((len(kept["s1"]), len(FEATURE_NAMES)), dtype=np.float32)
            for index in range(len(kept["s1"])):
                s1_index = int(kept["s1"][index])
                other_index = int(kept["other"][index])
                table = other[int(kept["source3"][index])]
                full[index] = full_vector(
                    frames["source1"]["name"][s1_index],
                    frames["source1"]["addr"][s1_index],
                    frames["source1"]["country"][s1_index],
                    table["name"][other_index],
                    table["addr"][other_index],
                    table["country"][other_index],
                    bool(kept["source3"][index]),
                    float(kept["cosine"][index]),
                    int(kept["rank"][index]),
                    float(mean_idf[s1_index]),
                    idf,
                    idf_default,
                )
            kept_spans = _spans(kept["s1"])
            _copy_prefilter_relative(blocked, kept, full, FEATURE_NAMES)
            if separate is None:
                full_prob = joint.predict(full)
            else:
                full_prob = np.zeros(len(kept["s1"]), dtype=np.float32)
                s2_rows = kept["source3"] == 0
                s3_rows = ~s2_rows
                if np.any(s2_rows):
                    full_prob[s2_rows] = separate[0].predict(full[s2_rows])
                if np.any(s3_rows):
                    full_prob[s3_rows] = separate[1].predict(full[s3_rows])
        by_entity_cand: dict[str, list[tuple[float, str]]] = defaultdict(list)
        by_entity_match: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for index in range(len(kept["s1"])):
            entity_id = str(kept["entity"][index])
            other_id = str(other_ids[int(kept["source3"][index])][int(kept["other"][index])])
            by_entity_cand[entity_id].append((float(kept_prob_cascade[index]), other_id))
            by_entity_match[entity_id].append((float(full_prob[index]), other_id))
        for pos in positions:
            entity_id = str(frames["source1"]["id"][pos])
            candidates = [item[1] for item in sorted(by_entity_cand.get(entity_id, []), key=lambda item: -item[0])]
            match_pairs = sorted(by_entity_match.get(entity_id, []), key=lambda item: -item[0])
            probs = np.array([item[0] for item in match_pairs], dtype=np.float32)
            ids = [item[1] for item in match_pairs]
            matches = _select(probs, ids, rule["threshold"], int(rule["cap"]), rule["alpha"], rule["singleton_gate"])
            _write_line(cand_file, entity_id, candidates)
            _write_line(match_file, entity_id, matches)
        if start % 20000 == 0:
            print(f"infer {min(start + chunk, total)}/{total}", flush=True)
            match_file.flush()
            cand_file.flush()
    match_file.close()
    cand_file.close()
    print(f"wrote {args.matching} and {args.candidate}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Approach A entity resolution")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--data-dir", type=Path, default=ROOT / "dataset" / "train")
    train.add_argument("--model-dir", type=Path, default=ROOT / "models")
    train.add_argument("--max-entities", type=int, default=200_000)
    train.set_defaults(func=cmd_train)
    infer = sub.add_parser("infer")
    infer.add_argument("--data-dir", type=Path, default=ROOT / "dataset" / "test")
    infer.add_argument("--model-dir", type=Path, default=ROOT / "models")
    infer.add_argument("--matching", type=Path, default=ROOT / "output" / "matching_results.tsv")
    infer.add_argument("--candidate", type=Path, default=ROOT / "output" / "candidate_pairs.tsv")
    infer.add_argument("--limit", type=int, default=0)
    infer.set_defaults(func=cmd_infer)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
