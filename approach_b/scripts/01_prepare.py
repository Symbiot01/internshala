"""Stage 1: clean and romanize every record once, and cache it as parquet.

Output: cache/<split>/source<N>.parquet with columns
id, country, name, address, name_roman, address_roman, plus mined
name_norm/address_norm/flags/parsed fields when tables exist.
Run scripts/09_enrich.py after 08_mine.py if tables arrive later.
"""

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
import data
import text

CHUNK = 50_000


def _prepare_chunk(pairs: tuple[list[str], list[str]]) -> list[tuple[str, str, str, str]]:
    names, addresses = pairs
    return [text.prepare_record(name, address) for name, address in zip(names, addresses)]


def prepared_path(split: str, source: int) -> Path:
    return config.CACHE_DIR / split / f"source{source}.parquet"


def prepare(split: str, source: int, pool: Pool) -> dict[str, int]:
    frame = data.read_source(split, source)
    names = frame.business_name.tolist()
    addresses = frame.business_address.tolist()
    chunks = [(names[i : i + CHUNK], addresses[i : i + CHUNK]) for i in range(0, len(names), CHUNK)]
    rows = [row for part in pool.map(_prepare_chunk, chunks) for row in part]
    out = pd.DataFrame(rows, columns=["name", "address", "name_roman", "address_roman"])
    out.insert(0, "country", frame.country.to_numpy())
    out.insert(0, "id", frame.entity_id.to_numpy())
    path = prepared_path(split, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    non_latin = int(out.name.map(text.has_non_latin_letters).sum())
    return {"rows": len(out), "non_latin_names": non_latin}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    args = parser.parse_args()
    report: dict[str, dict] = {}
    with Pool(config.WORKERS) as pool:
        for split in args.splits:
            for source in (1, 2, 3):
                start = time.time()
                stats = prepare(split, source, pool)
                stats["seconds"] = round(time.time() - start, 1)
                report[f"{split}_source{source}"] = stats
                print(split, source, stats, flush=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "01_prepare.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
