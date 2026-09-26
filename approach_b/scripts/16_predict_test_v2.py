"""Isolated v2 test scoring. Does not replace scripts/05_predict_test.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import calibrate
import claimants
import config
import cross_encoder
import decide
import distributed
import pairs as pair_tables
import pairs_v2
import stacker_v2


def write_report(name: str, report: dict) -> None:
    if not distributed.is_main():
        return
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / f"{name}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def score_shards(table: pd.DataFrame, frames, idf, include_country: bool, device, rank: int, world: int) -> pd.DataFrame | None:
    tokenizer, model = cross_encoder.load_checkpoint(config.MODELS_DIR / "matcher", device=device)
    shard_dir = config.CACHE_DIR / "test" / "v2" / "scores"
    shard_dir.mkdir(parents=True, exist_ok=True)
    entities = table.s1_id.drop_duplicates().to_numpy()
    for index, start in enumerate(range(0, len(entities), config.SCORE_SHARD_ENTITIES)):
        if index % world != rank:
            continue
        chunk_ids = set(entities[start : start + config.SCORE_SHARD_ENTITIES].tolist())
        path = shard_dir / f"shard_{start:07d}.parquet"
        if path.exists():
            print(f"skip {path.name}", flush=True)
            continue
        chunk = table[table.s1_id.isin(chunk_ids)].reset_index(drop=True)
        logits = cross_encoder.predict_logits(
            model, tokenizer, *cross_encoder.pair_texts(chunk, frames, include_country=include_country, idf=idf),
            device=device,
        )
        out = chunk[["s1_id", "cand_id"]].copy()
        out["logit"] = logits
        out.to_parquet(path, index=False)
        print(f"wrote {path.name} ({len(out)} pairs)", flush=True)
    distributed.barrier()
    if rank != 0:
        return None
    pieces = [pd.read_parquet(path) for path in sorted(shard_dir.glob("shard_*.parquet"))]
    return pd.concat(pieces, ignore_index=True)


def write_tsv(path: Path, entities: np.ndarray, mapping: dict[str, list[str]], column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"source1_entity_id\t{column}\n")
        for entity in entities:
            handle.write(f"{entity}\t{','.join(mapping.get(entity, []))}\n")


def main() -> None:
    distributed.maybe_relaunch()
    rank, world, device = distributed.setup()
    decision = json.loads((config.REPORTS_DIR / "15_decision.json").read_text())
    if decision.get("score_source") == "keep_04":
        raise SystemExit("v2 gate failed; keep 04 candidates")
    rule = decision["rule"]
    temperature = float(decision["temperature"])
    include_country = json.loads((config.REPORTS_DIR / "04_backbone.json").read_text()).get("include_country", True)
    if not (config.CACHE_DIR / "test" / "v2" / "candidates_v2_s2.npz").exists():
        raise SystemExit("test v2 candidates missing")
    if rank == 0 and not pairs_v2.pair_path("test").exists():
        write_report("16_pairs_test_v2", pairs_v2.write_pairs("test"))
    distributed.barrier()
    table = pairs_v2.load_pairs("test")
    frames = cross_encoder.lookup_frames("test")
    idf = cross_encoder.address_idf([pair_tables.load_prepared("test", source) for source in (1, 2, 3)])
    scored = score_shards(table, frames, idf, include_country, device, rank, world)
    if rank != 0:
        distributed.cleanup()
        return
    probs = calibrate.probabilities(scored.logit.to_numpy(), temperature)
    s1 = pair_tables.load_prepared("test", 1)
    others = {"S2": pair_tables.load_prepared("test", 2), "S3": pair_tables.load_prepared("test", 3)}
    index = claimants.ClaimantIndex(s1)
    packed_tmp = decide.pack(scored.s1_id.to_numpy(), scored.cand_id.to_numpy(), probs)
    feats = claimants.pair_features(packed_tmp["s1_ids"], packed_tmp["cand_ids"], s1, others, index)
    lookups = {}
    for source, frame in ((2, others["S2"]), (3, others["S3"])):
        payload = np.load(config.CACHE_DIR / "test" / "v2" / f"reverse_lookup_s{source}.npz")
        lookups[source] = (payload["positions"], payload["scores"], {i: p for p, i in enumerate(frame.id.to_numpy())})
    feats.update(stacker_v2.reverse_features(packed_tmp["s1_ids"], packed_tmp["cand_ids"], s1, lookups))
    x = stacker_v2.matrix(packed_tmp["scores"], packed_tmp["ranks_source"], packed_tmp["maxima"], feats)
    probs = stacker_v2.predict(stacker_v2.load(), x)
    packed = decide.pack(packed_tmp["s1_ids"], packed_tmp["cand_ids"], probs)
    triples = decide.select(packed, rule["threshold"], rule["cap"], rule["alpha"], rule["gate"])
    matches = decide.unique_mapping(triples)
    s1_ids = s1.id.to_numpy()
    candidates: dict[str, list[str]] = {entity: [] for entity in s1_ids}
    for entity, cand_id in zip(table.s1_id.to_numpy(), table.cand_id.to_numpy()):
        candidates[entity].append(str(cand_id))
    out_dir = config.OUTPUT_DIR / "v2"
    matching_path = out_dir / "matching_results.tsv"
    candidate_path = out_dir / "candidate_pairs.tsv"
    write_tsv(matching_path, s1_ids, matches, "matched_entity_ids")
    write_tsv(candidate_path, s1_ids, candidates, "candidate_entity_ids")
    command = [
        sys.executable,
        str(config.VALIDATOR),
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(config.DATASET_DIR / "test"),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    write_report("16_predict_test_v2", {
        "entities": int(len(s1_ids)),
        "candidate_pairs": int(len(table)),
        "matched_pairs": int(sum(len(v) for v in matches.values())),
        "validator_returncode": result.returncode,
        "validator_stdout": result.stdout,
        "validator_stderr": result.stderr,
        "rule": rule,
        "fallback_phase0_exists": (config.OUTPUT_DIR / "fallback_phase0" / "matching_results.tsv").exists(),
    })
    distributed.cleanup()
    if result.returncode != 0:
        raise SystemExit(result.stdout + result.stderr)


if __name__ == "__main__":
    main()
