"""Write isolated v2 sidecar parquets. Does not touch cache/*/source*.parquet."""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
import text
import translit

WORKERS = 32
CHUNK = 20_000
_STATE = None


def _init(kind: str, vocab_path: str) -> None:
    global _STATE
    dictionary, table = load_tables(kind)
    vocab = Path(vocab_path).read_text().splitlines()
    snapper = translit.VocabSnapper(vocab) if vocab else None
    _STATE = (dictionary, table, snapper, {})


def _chunk(names: list[str], romans: list[str], addresses: list[str], ids: list[str]) -> pd.DataFrame:
    dictionary, table, snapper, cache = _STATE
    rows = []
    for entity, name, roman, address in zip(ids, names, romans, addresses):
        indic = bool(text.INDIC_CHAR.search(name or ""))
        if indic:
            name_norm = translit.romanize_name(name, dictionary, table, snapper, cache)
        else:
            name_norm = " ".join(translit.latin_tokens(roman or name or ""))
        retrieval = ", ".join(p for p in (name_norm, roman or "", address or "") if p)
        rows.append({
            "id": entity,
            "name_norm_v2": name_norm,
            "retrieval_v2": retrieval,
            "name_skel": translit.skeleton_key(name_norm or roman or ""),
            "is_indic": "1" if indic else "0",
        })
    return pd.DataFrame(rows)


def load_tables(kind: str) -> tuple[dict, dict]:
    folder = config.MODELS_DIR / "tables_v2" / kind
    dictionary = json.loads((folder / "indic_dict.json").read_text())
    table = json.loads((folder / "m2m.json").read_text())
    return dictionary, table


def enrich(split: str, kind: str) -> dict:
    dictionary, table = load_tables(kind)
    out_dir = config.CACHE_DIR / split / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    s1 = pd.read_parquet(config.CACHE_DIR / split / "source1.parquet", columns=["name_roman"])
    vocab = sorted({tok for name in s1.name_roman.tolist() for tok in translit.latin_tokens(str(name)) if len(tok) >= 2})
    vocab_path = config.MODELS_DIR / "tables_v2" / kind / "s1_vocab.txt"
    vocab_path.write_text("\n".join(vocab))
    report = {}
    with Pool(WORKERS, initializer=_init, initargs=(kind, str(vocab_path))) as pool:
        for source in (1, 2, 3):
            base = pd.read_parquet(config.CACHE_DIR / split / f"source{source}.parquet", columns=["id", "name", "name_roman", "address"])
            payloads = []
            for i in range(0, len(base), CHUNK):
                part = base.iloc[i : i + CHUNK]
                payloads.append((
                    part.name.tolist(),
                    part.name_roman.tolist(),
                    part.address.tolist(),
                    part.id.tolist(),
                ))
            pieces = pool.starmap(_chunk, payloads)
            out = pd.concat(pieces, ignore_index=True)
            path = out_dir / f"source{source}.parquet"
            out.to_parquet(path, index=False)
            stats = {"rows": len(out), "indic": int((out.is_indic == "1").sum())}
            report[f"{split}_source{source}"] = stats
            print(split, source, stats, flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    args = parser.parse_args()
    report = {}
    for split in args.splits:
        kind = "measure" if split == "train" else "ship"
        report.update(enrich(split, kind))
    (config.MODELS_DIR / "tables_v2" / "sidecar.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
