"""Mine transliteration, alias, and unigram tables from provided training gold.

Hold-out Source 1 ids are excluded so later eval is honest. Tables are written
as JSON with a provenance sidecar. No network, no external gazetteers.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import config
import data
import text

SCRIPT_RANGES = (
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("oriya", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
)


def script_of(value: str) -> str:
    for char in value:
        code = ord(char)
        for name, start, end in SCRIPT_RANGES:
            if start <= code <= end:
                return name
    return "none"


def _holdout_s1_ids() -> set[str]:
    groups = np.load(config.CACHE_DIR / "train" / "entities.npz")
    s1 = pd.read_parquet(config.CACHE_DIR / "train" / "source1.parquet", columns=["id"])
    hold = np.concatenate([groups["calibration"], groups["evaluation"]])
    return set(s1.id.to_numpy()[hold].tolist())


def _load_sources() -> dict[str, pd.DataFrame]:
    frames = {}
    for source, key in ((1, "S1"), (2, "S2"), (3, "S3")):
        frames[key] = pd.read_parquet(config.CACHE_DIR / "train" / f"source{source}.parquet")
    return frames


def mine_indic_dict(pairs: list[tuple[str, str]], min_support: int = 2, min_purity: float = 0.5) -> dict[str, str]:
    import difflib

    positional: dict[str, Counter] = defaultdict(Counter)
    em_pairs: list[tuple[list[str], list[str]]] = []
    for latin, native in pairs:
        src = text.tokens(text.expand_tokens(latin))
        tgt = native.split()
        indic = [tok for tok in tgt if text.INDIC_CHAR.search(tok)]
        latin_keep = [tok for tok in src if tok not in text.LEGAL]
        if not indic or not latin_keep:
            continue
        if len(indic) == len(latin_keep):
            for a, b in zip(indic, latin_keep):
                positional[a][b] += 1
            continue
        for a in indic:
            scored = max(
                latin_keep,
                key=lambda w: difflib.SequenceMatcher(
                    None, text.phonetic_skeleton(text.brahmic_romanize(a)), text.phonetic_skeleton(w)
                ).ratio(),
            )
            ratio = difflib.SequenceMatcher(
                None, text.phonetic_skeleton(text.brahmic_romanize(a)), text.phonetic_skeleton(scored)
            ).ratio()
            if ratio >= 0.4:
                positional[a][scored] += 1
            else:
                em_pairs.append((latin_keep, [a]))
    table = _model1(em_pairs) if em_pairs else {}
    combined: dict[str, Counter] = defaultdict(Counter)
    for native, counts in positional.items():
        combined[native].update(counts)
    for native, latin_map in table.items():
        for latin, weight in latin_map.items():
            combined[native][latin] += max(int(weight), 1)
    out = {}
    for native, counts in combined.items():
        latin, support = counts.most_common(1)[0]
        total = sum(counts.values())
        if support >= min_support and support / total >= min_purity:
            out[native] = latin
    return out


def _model1(pairs: list[tuple[list[str], list[str]]], iterations: int = 6) -> dict[str, Counter]:
    trans: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 1.0))
    for _ in range(iterations):
        count: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        total_src: dict[str, float] = defaultdict(float)
        for src, tgt in pairs:
            for native in tgt:
                z = sum(trans[native][latin] for latin in src) or 1.0
                for latin in src:
                    c = trans[native][latin] / z
                    count[native][latin] += c
                    total_src[latin] += c
        trans = defaultdict(lambda: defaultdict(float))
        for native, latin_map in count.items():
            for latin, c in latin_map.items():
                trans[native][latin] = c / max(total_src[latin], 1e-9)
    out: dict[str, Counter] = {}
    for native, latin_map in trans.items():
        if latin_map:
            out[native] = Counter({k: int(v * 1000) for k, v in latin_map.items()})
    return out


def mine_aliases(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Map last/second-last address segments of gold records onto the S1 form."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for s1_addr, other_addr in pairs:
        s1_parts = [p.strip() for p in s1_addr.split(",") if p.strip()]
        other_parts = [p.strip() for p in other_addr.split(",") if p.strip()]
        if not s1_parts or not other_parts:
            continue
        for s1_seg, other_seg in ((s1_parts[-1], other_parts[-1]),):
            if s1_seg and other_seg and s1_seg != other_seg:
                counts[other_seg][s1_seg] += 1
        if len(s1_parts) >= 2 and len(other_parts) >= 2:
            counts[other_parts[-2]][s1_parts[-2]] += 1
    aliases = dict(text.SEED_ALIASES)
    for src, dests in counts.items():
        dest, n = dests.most_common(1)[0]
        if n >= 8 and n / sum(dests.values()) >= 0.6 and src != dest:
            aliases[src] = dest
            aliases[text.romanize(src)] = dest
    return aliases


def mine_unigrams(names: list[str]) -> tuple[dict[str, int], int]:
    counts: Counter[str] = Counter()
    for name in names:
        counts.update(tok for tok in text.tokens(name) if len(tok) >= 2)
    return dict(counts), int(sum(counts.values()))


def evaluate_indic_dict(dictionary: dict[str, str], pairs: list[tuple[str, str]]) -> dict:
    by_script: dict[str, list[float]] = defaultdict(list)
    covered = total = 0
    for latin, native in pairs:
        mapped = text.apply_indic_dict(native, dictionary)
        jac = _char_jaccard(latin, mapped)
        by_script[script_of(native)].append(jac)
        for tok in native.split():
            if text.INDIC_CHAR.search(tok):
                total += 1
                covered += int(tok in dictionary)
    summary = {
        "pairs": len(pairs),
        "token_coverage": covered / max(total, 1),
        "mean_jaccard": float(np.mean([x for xs in by_script.values() for x in xs]) if pairs else 0),
        "per_script": {
            name: {"n": len(xs), "mean_jaccard": float(np.mean(xs))}
            for name, xs in sorted(by_script.items(), key=lambda kv: -len(kv[1]))
        },
    }
    return summary


def _char_jaccard(a: str, b: str) -> float:
    def grams(s: str) -> set[str]:
        s = " " + " ".join(text.tokens(s)) + " "
        return {s[i:i + 3] for i in range(max(len(s) - 2, 0))}
    A, B = grams(a), grams(b)
    return len(A & B) / max(len(A | B), 1)


def run(split: str = "train") -> dict:
    frames = _load_sources()
    holdout = _holdout_s1_ids()
    truth = data.read_truth("train")
    id_index = {
        key: dict(zip(frame.id.to_numpy(), range(len(frame))))
        for key, frame in frames.items()
    }
    train_indic: list[tuple[str, str]] = []
    hold_indic: list[tuple[str, str]] = []
    train_addr: list[tuple[str, str]] = []
    for s1_id, gold in truth.items():
        if s1_id not in id_index["S1"]:
            continue
        s1_row = id_index["S1"][s1_id]
        s1_name = frames["S1"].name_roman.iat[s1_row]
        s1_addr = frames["S1"].address_roman.iat[s1_row]
        dest = hold_indic if s1_id in holdout else train_indic
        for gid in gold:
            src = gid[:2]
            if gid not in id_index[src]:
                continue
            row = id_index[src][gid]
            other_name = frames[src].name.iat[row]
            other_addr = frames[src].address.iat[row]
            if text.INDIC_CHAR.search(other_name):
                dest.append((s1_name, other_name))
            if s1_id not in holdout:
                train_addr.append((s1_addr, text.romanize(text.apply_indic_dict(other_addr, {}))))
    dictionary = mine_indic_dict(train_indic)
    aliases = mine_aliases(train_addr)
    unigrams, total = mine_unigrams(frames["S1"].name_roman.tolist())
    hold_report = evaluate_indic_dict(dictionary, hold_indic)
    folder = config.TABLES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    _write(folder / "indic_dict.json", dictionary)
    _write(folder / "aliases.json", aliases)
    _write(folder / "unigrams.json", {"total": total, "counts": unigrams})
    _write(folder / "french_streets.json", text.FRENCH_STREET)
    provenance = {
        "source": "train gold pairs whose Source 1 id is outside calibration+evaluation",
        "indic_train_pairs": len(train_indic),
        "indic_hold_pairs": len(hold_indic),
        "dict_size": len(dictionary),
        "alias_size": len(aliases),
        "unigram_types": len(unigrams),
        "holdout_eval": hold_report,
        "fair_play": "tables are derived only from provided training records",
    }
    _write(folder / "provenance.json", provenance)
    return provenance


def load_tables(folder: Path | None = None) -> dict:
    folder = folder or config.TABLES_DIR
    indic_path = folder / "indic_dict.json"
    alias_path = folder / "aliases.json"
    uni_path = folder / "unigrams.json"
    dictionary = json.loads(indic_path.read_text()) if indic_path.exists() else {}
    aliases = json.loads(alias_path.read_text()) if alias_path.exists() else dict(text.SEED_ALIASES)
    unigrams, total = {}, 1
    if uni_path.exists():
        payload = json.loads(uni_path.read_text())
        unigrams, total = payload.get("counts", {}), int(payload.get("total") or 1)
    return {"indic_dict": dictionary, "aliases": aliases, "unigrams": unigrams, "unigram_total": total}


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
