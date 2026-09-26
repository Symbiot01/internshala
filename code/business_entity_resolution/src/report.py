"""Score five disjoint training hold-outs the matcher never trained on.

Uses only train ground truth. No portal upload and no external APIs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.data.io import parse_id_list
from src.eval.metrics import macro_prf
from src.matching.features import CHEAP_NAMES, FEATURE_NAMES, cheap_vector, full_vector
from src.pipeline import (
    ROOT,
    _append_source,
    _concat,
    _copy_prefilter_relative,
    _idf_lookup,
    _keep_mask,
    _mean_idf,
    _prepare_split,
    _select,
    _slice,
    _sort_by_entity,
    _spans,
    sample_positions,
)

SET_SIZE = 2000
N_SETS = 5


def _clean(value: str) -> str:
    return " ".join(str(value).replace("\t", " ").replace("\n", " ").split())


def _lookup_tables(split: dict) -> dict[str, tuple[str, str, str]]:
    rows: dict[str, tuple[str, str, str]] = {}
    for name in ("source1", "source2", "source3"):
        frame = split[name]
        for row in frame.itertuples(index=False):
            rows[row.entity_id] = (row.business_name, row.business_address, row.country)
    return rows


def _draw_sets(s1_ids: np.ndarray, countries: np.ndarray, counts: dict[str, int]) -> list[np.ndarray]:
    trained = set(int(pos) for pos in sample_positions(s1_ids, countries, counts, 200_000, seed=0))
    pool = np.array([index for index in range(len(s1_ids)) if index not in trained], dtype=np.int64)
    sets: list[np.ndarray] = []
    for seed in range(1, N_SETS + 1):
        local = sample_positions(s1_ids[pool], countries[pool], counts, SET_SIZE, seed=seed)
        chosen = pool[local]
        sets.append(chosen)
        keep = np.ones(len(pool), dtype=bool)
        keep[local] = False
        pool = pool[keep]
        print(f"set {seed}: {len(chosen)} entities remaining_pool={len(pool)}", flush=True)
    return sets


def _score_positions(
    positions: np.ndarray,
    frames,
    indexes,
    gold: dict[str, set[str]],
    cascade,
    joint,
    config: dict,
    idf: dict[str, float],
    idf_default: float,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    other = {0: frames["source2"], 1: frames["source3"]}
    other_ids = {0: frames["source2"]["id"], 1: frames["source3"]["id"]}
    mean_idf = np.zeros(len(frames["source1"]["id"]), dtype=np.float32)
    for pos in positions:
        mean_idf[pos] = _mean_idf(frames["source1"]["name"][pos], idf, idf_default)
    parts = [
        _append_source(
            positions,
            frames["source1"]["name"],
            frames["source1"]["addr"],
            indexes["source2"],
            False,
            gold,
            frames["source1"]["id"],
        ),
        _append_source(
            positions,
            frames["source1"]["name"],
            frames["source1"]["addr"],
            indexes["source3"],
            True,
            gold,
            frames["source1"]["id"],
        ),
    ]
    predicted: dict[str, set[str]] = {
        str(frames["source1"]["id"][pos]): set() for pos in positions
    }
    blocked_ids: dict[str, set[str]] = {key: set() for key in predicted}
    candidates: dict[str, set[str]] = {key: set() for key in predicted}
    if len(parts[0]["s1"]) == 0 and len(parts[1]["s1"]) == 0:
        return predicted, blocked_ids, candidates
    blocked = _sort_by_entity(_concat(parts))
    blocked["entity"] = frames["source1"]["id"][blocked["s1"]]
    cheap = np.zeros((len(blocked["s1"]), len(CHEAP_NAMES)), dtype=np.float32)
    for index in range(len(blocked["s1"])):
        s1_index = int(blocked["s1"][index])
        other_index = int(blocked["other"][index])
        table = other[int(blocked["source3"][index])]
        other_id = str(other_ids[int(blocked["source3"][index])][other_index])
        blocked_ids[str(blocked["entity"][index])].add(other_id)
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
    if len(kept["s1"]) == 0:
        return predicted, blocked_ids, candidates
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
    _copy_prefilter_relative(blocked, kept, full, FEATURE_NAMES)
    full_prob = joint.predict(full)
    by_entity: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for index in range(len(kept["s1"])):
        entity_id = str(kept["entity"][index])
        other_id = str(other_ids[int(kept["source3"][index])][int(kept["other"][index])])
        candidates[entity_id].add(other_id)
        by_entity[entity_id].append((float(full_prob[index]), other_id))
    for entity_id, pairs in by_entity.items():
        pairs.sort(key=lambda item: -item[0])
        probs = np.array([item[0] for item in pairs], dtype=np.float32)
        ids = [item[1] for item in pairs]
        predicted[entity_id] = set(
            _select(
                probs,
                ids,
                config["threshold"],
                int(config["cap"]),
                config["alpha"],
                config["singleton_gate"],
            )
        )
    return predicted, blocked_ids, candidates


def _pair_counts(predicted: dict[str, set[str]], truth: dict[str, set[str]]) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for entity_id, gold in truth.items():
        pred = predicted.get(entity_id, set())
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    return tp, fp, fn


def _errors_for_set(
    set_id: int,
    predicted: dict[str, set[str]],
    truth: dict[str, set[str]],
    blocked: dict[str, set[str]],
    candidates: dict[str, set[str]],
    records: dict[str, tuple[str, str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for entity_id, gold in truth.items():
        pred = predicted.get(entity_id, set())
        left = records.get(entity_id, ("", "", ""))
        blocked_ids = blocked.get(entity_id, set())
        cand_ids = candidates.get(entity_id, set())
        if not gold and pred:
            for other_id in sorted(pred):
                right = records.get(other_id, ("", "", ""))
                rows.append(
                    {
                        "set_id": str(set_id),
                        "source1_entity_id": entity_id,
                        "other_entity_id": other_id,
                        "error_type": "singleton_false_merge",
                        "country": left[2],
                        "s1_name": _clean(left[0]),
                        "s1_address": _clean(left[1]),
                        "other_name": _clean(right[0]),
                        "other_address": _clean(right[1]),
                        "in_candidates": "1" if other_id in cand_ids else "0",
                    }
                )
            continue
        for other_id in sorted(pred - gold):
            right = records.get(other_id, ("", "", ""))
            rows.append(
                {
                    "set_id": str(set_id),
                    "source1_entity_id": entity_id,
                    "other_entity_id": other_id,
                    "error_type": "false_positive",
                    "country": left[2],
                    "s1_name": _clean(left[0]),
                    "s1_address": _clean(left[1]),
                    "other_name": _clean(right[0]),
                    "other_address": _clean(right[1]),
                    "in_candidates": "1" if other_id in cand_ids else "0",
                }
            )
        for other_id in sorted(gold - pred):
            right = records.get(other_id, ("", "", ""))
            error_type = "blocking_miss" if other_id not in blocked_ids else "false_negative"
            rows.append(
                {
                    "set_id": str(set_id),
                    "source1_entity_id": entity_id,
                    "other_entity_id": other_id,
                    "error_type": error_type,
                    "country": left[2],
                    "s1_name": _clean(left[0]),
                    "s1_address": _clean(left[1]),
                    "other_name": _clean(right[0]),
                    "other_address": _clean(right[1]),
                    "in_candidates": "1" if other_id in cand_ids else "0",
                }
            )
    return rows


def _example_block(rows: list[dict[str, str]], error_type: str, limit: int = 15) -> str:
    picked = [row for row in rows if row["error_type"] == error_type][:limit]
    if not picked:
        return f"No {error_type} rows.\n"
    lines = [f"### {error_type} ({len(picked)} shown)\n"]
    for row in picked:
        lines.append(
            f"- set {row['set_id']} `{row['source1_entity_id']}` vs `{row['other_entity_id']}` "
            f"({row['country']}, in_candidates={row['in_candidates']})\n"
            f"  S1: {row['s1_name']} | {row['s1_address']}\n"
            f"  Other: {row['other_name']} | {row['other_address']}"
        )
    return "\n".join(lines) + "\n"


def _write_markdown(path: Path, scores: list[dict], errors: list[dict[str, str]]) -> None:
    f05 = [row["macro_f05"] for row in scores]
    type_counts = Counter(row["error_type"] for row in errors)
    country_counts = Counter((row["error_type"], row["country"]) for row in errors)
    lines = [
        "# Hold-out error report",
        "",
        "Five disjoint 2,000-entity slices of training Source 1. None of these ids are in the seed-0 200k training sample. Decision rule is the saved matcher: threshold 0.7, cap 11, relative margin 0.75, singleton gate 0.6.",
        "",
        f"Mean F0.5 {sum(f05) / len(f05):.4f}. Min {min(f05):.4f}. Max {max(f05):.4f}.",
        "",
        "| set | entities | singletons | macro P | macro R | macro F0.5 | TP | FP | FN | blocking_miss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scores:
        lines.append(
            f"| {row['set_id']} | {int(row['entities'])} | {int(row['singletons'])} | "
            f"{row['macro_precision']:.4f} | {row['macro_recall']:.4f} | {row['macro_f05']:.4f} | "
            f"{row['tp']} | {row['fp']} | {row['fn']} | {row['blocking_miss']} |"
        )
    lines.extend(
        [
            "",
            "## Error counts by type",
            "",
            "| type | count |",
            "|---|---:|",
        ]
    )
    for name in ("false_positive", "false_negative", "blocking_miss", "singleton_false_merge"):
        lines.append(f"| {name} | {type_counts.get(name, 0)} |")
    lines.extend(["", "## Error counts by type and country", "", "| type | country | count |", "|---|---|---:|"])
    for (error_type, country), count in sorted(country_counts.items()):
        lines.append(f"| {error_type} | {country} | {count} |")
    lines.extend(["", "## Examples", ""])
    for error_type in ("false_positive", "false_negative", "blocking_miss", "singleton_false_merge"):
        lines.append(_example_block(errors, error_type))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset" / "train")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    config = json.loads((args.model_dir / "config.json").read_text())
    cascade = lgb.Booster(model_file=str(args.model_dir / "cascade.txt"))
    joint = lgb.Booster(model_file=str(args.model_dir / "matcher.txt"))
    split, frames, indexes = _prepare_split(args.data_dir, args.model_dir / "cache")
    records = _lookup_tables(split)
    gold_all: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for row in split["ground_truth"].itertuples(index=False):
        parsed = set(parse_id_list(row.matched_entity_ids))
        gold_all[row.source1_entity_id] = parsed
        counts[row.source1_entity_id] = len(parsed)
    s1_ids = frames["source1"]["id"]
    countries = frames["source1"]["country"]
    print("drawing hold-out sets", flush=True)
    sets = _draw_sets(s1_ids, countries, counts)
    idf, idf_default = _idf_lookup(indexes["source2"])
    idf.update(_idf_lookup(indexes["source3"])[0])
    scores: list[dict] = []
    errors: list[dict[str, str]] = []
    for set_id, positions in enumerate(sets, start=1):
        print(f"scoring set {set_id}", flush=True)
        gold = {str(s1_ids[pos]): gold_all.get(str(s1_ids[pos]), set()) for pos in positions}
        predicted, blocked, candidates = _score_positions(
            positions, frames, indexes, gold, cascade, joint, config, idf, idf_default
        )
        metrics = macro_prf(predicted, gold)
        tp, fp, fn = _pair_counts(predicted, gold)
        set_errors = _errors_for_set(set_id, predicted, gold, blocked, candidates, records)
        blocking_miss = sum(1 for row in set_errors if row["error_type"] == "blocking_miss")
        scores.append(
            {
                "set_id": set_id,
                "entities": metrics["entities"],
                "singletons": metrics["singletons"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f05": metrics["macro_f05"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "blocking_miss": blocking_miss,
            }
        )
        errors.extend(set_errors)
        print(
            f"set {set_id} f05={metrics['macro_f05']:.4f} "
            f"P={metrics['macro_precision']:.4f} R={metrics['macro_recall']:.4f} "
            f"tp={tp} fp={fp} fn={fn} blocking_miss={blocking_miss}",
            flush=True,
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    error_path = args.out_dir / "errors.tsv"
    fields = [
        "set_id",
        "source1_entity_id",
        "other_entity_id",
        "error_type",
        "country",
        "s1_name",
        "s1_address",
        "other_name",
        "other_address",
        "in_candidates",
    ]
    with error_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(errors)
    report_path = args.out_dir / "holdout_report.md"
    _write_markdown(report_path, scores, errors)
    print(f"wrote {error_path} ({len(errors)} rows) and {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
