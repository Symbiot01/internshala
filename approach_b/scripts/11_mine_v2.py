"""Mine v2 Indic tables. Writes only under models/tables_v2/."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config
import data
import mining
import text
import translit

ROOT = config.MODELS_DIR / "tables_v2"


def _pairs(frames: dict[str, pd.DataFrame], holdout: set[str] | None, keep_holdout: bool = False) -> list[tuple[str, str]]:
    latin = dict(zip(frames["S1"].id.to_numpy(), frames["S1"].name_roman.to_numpy()))
    natives = {
        **dict(zip(frames["S2"].id.to_numpy(), frames["S2"].name.to_numpy())),
        **dict(zip(frames["S3"].id.to_numpy(), frames["S3"].name.to_numpy())),
    }
    out = []
    for s1_id, gold in data.read_truth("train").items():
        if keep_holdout:
            if s1_id not in holdout:
                continue
        elif holdout is not None and s1_id in holdout:
            continue
        src_name = latin.get(s1_id)
        if src_name is None:
            continue
        for gid in gold:
            native = natives.get(gid)
            if native is not None and text.INDIC_CHAR.search(str(native)):
                out.append((str(src_name), str(native)))
    return out


def _word_pairs(name_pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    words = []
    for latin, native in name_pairs:
        src = translit.latin_tokens(latin)
        tgt = translit.native_tokens(native)
        if len(src) != len(tgt):
            continue
        for a, b in zip(tgt, src):
            if text.INDIC_CHAR.search(a) and b:
                words.append((a, b))
    return words


def _eval(dictionary, table, snapper, hold_pairs: list[tuple[str, str]]) -> dict:
    by_script: dict[str, list[float]] = defaultdict(list)
    exact = total = 0
    cache: dict[str, str] = {}
    for latin, native in hold_pairs:
        mapped = translit.romanize_name(native, dictionary, table, snapper, cache)
        a, b = set(translit.latin_tokens(latin)), set(translit.latin_tokens(mapped))
        jac = len(a & b) / max(len(a | b), 1)
        by_script[mining.script_of(native)].append(jac)
        src = translit.latin_tokens(latin)
        tgt = translit.native_tokens(native)
        if len(src) == len(tgt):
            for a, b in zip(tgt, src):
                if text.INDIC_CHAR.search(a):
                    total += 1
                    exact += int(cache.get(a) == b)
    overall = [x for xs in by_script.values() for x in xs]
    return {
        "pairs": len(hold_pairs),
        "mean_jaccard": float(np.mean(overall) if overall else 0),
        "exact_token_accuracy": exact / max(total, 1),
        "dict_size": len(dictionary),
        "m2m_units": len(table),
        "per_script": {
            name: {"n": len(xs), "mean_jaccard": float(np.mean(xs))}
            for name, xs in sorted(by_script.items(), key=lambda kv: -len(kv[1]))
        },
    }


def _dump(folder: Path, dictionary, table, report) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "indic_dict.json").write_text(json.dumps(dictionary, ensure_ascii=False, indent=2))
    (folder / "m2m.json").write_text(json.dumps(table, ensure_ascii=False))
    (folder / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    frames = {
        "S1": pd.read_parquet(config.CACHE_DIR / "train" / "source1.parquet", columns=["id", "name_roman"]),
        "S2": pd.read_parquet(config.CACHE_DIR / "train" / "source2.parquet", columns=["id", "name"]),
        "S3": pd.read_parquet(config.CACHE_DIR / "train" / "source3.parquet", columns=["id", "name"]),
    }
    holdout = mining._holdout_s1_ids()
    print("collecting pairs", flush=True)
    train_pairs = _pairs(frames, holdout, keep_holdout=False)
    hold_pairs = _pairs(frames, holdout, keep_holdout=True)
    print(f"train_pairs={len(train_pairs)} hold_pairs={len(hold_pairs)}", flush=True)

    print("positional mine", flush=True)
    dictionary = translit.mine_positional(train_pairs)
    words = _word_pairs(train_pairs)
    print(f"dict_size={len(dictionary)} word_pairs={len(words)}", flush=True)
    print("train m2m", flush=True)
    table = translit.train_m2m([(k, v) for k, v in dictionary.items()] * 4)
    hold_set = set(holdout)
    hold_mask = np.array([i in hold_set for i in frames["S1"].id.to_numpy()])
    print("build snapper", flush=True)
    measure_vocab = list(
        dict.fromkeys(
            tok
            for name, skip in zip(frames["S1"].name_roman.to_numpy(), hold_mask)
            if not skip
            for tok in translit.latin_tokens(str(name))
        )
    )
    snapper = translit.VocabSnapper(measure_vocab)
    print("evaluate hold-out", flush=True)
    report = _eval(dictionary, table, snapper, hold_pairs)
    report["train_pairs"] = len(train_pairs)
    report["word_pairs"] = len(words)
    report["m2m_train_pairs"] = len(dictionary)
    tamil = report["per_script"].get("tamil", {}).get("mean_jaccard", 0)
    malayalam = report["per_script"].get("malayalam", {}).get("mean_jaccard", 0)
    report["gate"] = {
        "tamil": tamil,
        "malayalam": malayalam,
        "overall": report["mean_jaccard"],
        "pass": bool(tamil >= 0.90 and malayalam >= 0.90 and report["mean_jaccard"] >= 0.95),
    }
    _dump(ROOT / "measure", dictionary, table, report)

    print("ship tables", flush=True)
    ship_dict = translit.mine_positional(_pairs(frames, None))
    ship_table = translit.train_m2m([(k, v) for k, v in ship_dict.items()] * 4)
    _dump(ROOT / "ship", ship_dict, ship_table, {"dict_size": len(ship_dict), "m2m_units": len(ship_table)})
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not report["gate"]["pass"]:
        print("WARNING: Jaccard gate failed; still writing tables for retrieval concat fallback", flush=True)


if __name__ == "__main__":
    main()
