"""France alias mining, synthetic hold-out, and test-side sanity checks.

Synthetic pairs are for diagnostics only. Fine-tuning on unlabeled test records
is off by default (fair-play disclosure would be required if enabled).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config
import text


def mine_admin_aliases(s1: pd.DataFrame, others: list[pd.DataFrame]) -> dict[str, str]:
    """Pivot on shared city strings: S1 last-but-one/last vs S2/S3 last segments."""
    city_to_s1: dict[str, Counter] = defaultdict(Counter)
    france = s1[s1.country == "France"]
    for addr in france.address_roman.to_numpy():
        parts = [p.strip() for p in str(addr).split(",") if p.strip()]
        if len(parts) >= 2:
            city_to_s1[parts[-2]][parts[-1]] += 1
    aliases = dict(text.SEED_ALIASES)
    for other in others:
        fr = other[other.country == "France"]
        for addr in fr.address_roman.to_numpy():
            parts = [p.strip() for p in str(addr).split(",") if p.strip()]
            if len(parts) >= 2:
                city, admin = parts[-2], parts[-1]
                if city in city_to_s1:
                    region, _ = city_to_s1[city].most_common(1)[0]
                    if admin != region:
                        aliases[admin] = region
    aliases.update({k: v for k, v in text.FRENCH_STREET.items()})
    return aliases


def corrupt(name: str, address: str, rng: np.random.Generator) -> tuple[str, str]:
    op = rng.integers(0, 6)
    if op == 0:
        return name, ""
    if op == 1:
        parts = [p.strip() for p in address.split(",") if p.strip()]
        if len(parts) >= 2:
            return name, ", ".join(parts[:-1])
        return name, address
    if op == 2:
        return text.ocr_perturb(name, rng), address
    if op == 3:
        toks = name.split()
        return " ".join(toks[: max(1, len(toks) - 1)]), address
    if op == 4:
        stem = "".join(ch for ch in name.lower() if ch.isalnum())
        return stem[:24] + ".com", address
    return name, address.replace("rue", "r.").replace("boulevard", "bd")


def synthetic_holdout(s1: pd.DataFrame, n: int = 20_000, seed: int = config.SEED) -> pd.DataFrame:
    france = s1[s1.country == "France"].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    take = france.sample(n=min(n, len(france)), random_state=seed)
    rows = []
    for row in take.itertuples(index=False):
        noisy_name, noisy_addr = corrupt(row.name_roman, row.address_roman, rng)
        rows.append({"id": row.id, "name": noisy_name, "address": noisy_addr, "orig_name": row.name_roman, "orig_addr": row.address_roman})
    return pd.DataFrame(rows)


def prediction_sanity(matching_path: Path, s1: pd.DataFrame) -> dict:
    pred = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    merged = pred.merge(s1[["id", "country"]], left_on="source1_entity_id", right_on="id")
    out = {}
    for country, group in merged.groupby("country"):
        empty = group.matched_entity_ids.eq("")
        sizes = group.matched_entity_ids.where(~empty, "").str.split(",").map(lambda x: 0 if x == [""] else len(x))
        out[country] = {
            "entities": int(len(group)),
            "singleton_rate": float(empty.mean()),
            "mean_matches": float(sizes.mean()),
        }
    return out


def main() -> None:
    s1 = pd.read_parquet(config.CACHE_DIR / "test" / "source1.parquet")
    s2 = pd.read_parquet(config.CACHE_DIR / "test" / "source2.parquet")
    s3 = pd.read_parquet(config.CACHE_DIR / "test" / "source3.parquet")
    aliases = mine_admin_aliases(s1, [s2, s3])
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    existing = {}
    alias_path = config.TABLES_DIR / "aliases.json"
    if alias_path.exists():
        existing = json.loads(alias_path.read_text())
    existing.update(aliases)
    alias_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    synth = synthetic_holdout(s1)
    synth_path = config.CACHE_DIR / "test" / "synthetic_france.parquet"
    synth.to_parquet(synth_path, index=False)
    matching = config.OUTPUT_DIR / "fallback_phase0" / "matching_results.tsv"
    if not matching.exists():
        matching = config.OUTPUT_DIR / "matching_results.tsv"
    sanity = prediction_sanity(matching, s1) if matching.exists() else {}
    report = {
        "aliases_added": len(aliases),
        "synthetic_rows": int(len(synth)),
        "synthetic_path": str(synth_path),
        "fine_tune_on_test": False,
        "fine_tune_reason": "disabled; would use unlabeled test inputs and must be disclosed",
        "prediction_sanity": sanity,
        "french_street_table": text.FRENCH_STREET,
    }
    (config.REPORTS_DIR / "10_france.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
