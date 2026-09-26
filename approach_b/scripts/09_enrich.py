"""Add name_norm / address parse columns to an existing prepared parquet."""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
import mining
import text

CHUNK = 20_000
_TABLES = None


def _init(tables: dict) -> None:
    global _TABLES
    _TABLES = tables


def _enrich_chunk(payload: tuple[list, list, list, list]) -> list[dict]:
    names, addresses, name_roman, address_roman = payload
    return [
        text.enrich_record(n, a, nr, ar, _TABLES)
        for n, a, nr, ar in zip(names, addresses, name_roman, address_roman)
    ]


def enrich_split(split: str, tables: dict, pool: Pool) -> dict:
    report = {}
    for source in (1, 2, 3):
        path = config.CACHE_DIR / split / f"source{source}.parquet"
        frame = pd.read_parquet(path)
        start = time.time()
        names = frame.name.tolist()
        addresses = frame.address.tolist()
        name_roman = frame.name_roman.tolist()
        address_roman = frame.address_roman.tolist()
        chunks = [
            (
                names[i : i + CHUNK],
                addresses[i : i + CHUNK],
                name_roman[i : i + CHUNK],
                address_roman[i : i + CHUNK],
            )
            for i in range(0, len(frame), CHUNK)
        ]
        rows = [row for part in pool.map(_enrich_chunk, chunks) for row in part]
        extra = pd.DataFrame(rows)
        out = pd.concat([frame.reset_index(drop=True), extra], axis=1)
        out.to_parquet(path, index=False)
        stats = {
            "rows": len(out),
            "blank_address": int((out.blank_address == "1").sum()),
            "domain_like": int((out.domain_like == "1").sum()),
            "nonce_like": int((out.nonce_like == "1").sum()),
            "seconds": round(time.time() - start, 1),
        }
        report[f"{split}_source{source}"] = stats
        print(split, source, stats, flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    args = parser.parse_args()
    tables = mining.load_tables()
    report = {"tables": {k: (len(v) if hasattr(v, "__len__") and k != "unigram_total" else v) for k, v in tables.items()}}
    with Pool(config.WORKERS, initializer=_init, initargs=(tables,)) as pool:
        for split in args.splits:
            report.update(enrich_split(split, tables, pool))
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "01_enrich.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
