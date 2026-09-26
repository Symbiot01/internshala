"""Stage 7: retrieve, score, decide, and write the two submission TSVs.

Score shards under cache/test/scores/ are skipped on restart. Matches are a
subset of candidate_pairs.tsv because Unique Mapping Clustering only deletes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import calibrate
import config
import cross_encoder
import decide
import distributed
import pairs as pair_tables


def write_report(name: str, report: dict) -> None:
    if not distributed.is_main():
        return
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / f"{name}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def load_decision() -> dict:
    for name in ("07_decision.json", "04_decision.json"):
        path = config.REPORTS_DIR / name
        if path.exists():
            payload = json.loads(path.read_text())
            if "rule" in payload:
                return payload
    raise SystemExit("no decision report found")


def score_shards(table: pd.DataFrame, frames, idf, include_country: bool, device, rank: int, world: int) -> pd.DataFrame | None:
    tokenizer, model = cross_encoder.load_checkpoint(config.MODELS_DIR / "matcher", device=device)
    shard_dir = config.CACHE_DIR / "test" / "scores"
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
    decision = load_decision()
    rule = decision["rule"]
    temperature = float(decision["temperature"])
    include_country = json.loads((config.REPORTS_DIR / "04_backbone.json").read_text()).get("include_country", True)

    if not (config.CACHE_DIR / "test" / "candidates_s2.npz").exists():
        raise SystemExit("test candidates missing — run 02_retrieve.py --split test --stage lexical/dense/fuse first")

    if rank == 0 and not pair_tables.pair_path("test").exists():
        stats = pair_tables.write_pairs("test")
        write_report("05_pairs_test", stats)
    distributed.barrier()

    table = pair_tables.load_pairs("test")
    frames = cross_encoder.lookup_frames("test")
    idf = cross_encoder.address_idf([pair_tables.load_prepared("test", source) for source in (1, 2, 3)])
    scored = score_shards(table, frames, idf, include_country, device, rank, world)
    if rank != 0:
        distributed.cleanup()
        return
    probs = calibrate.probabilities(scored.logit.to_numpy(), temperature)
    stacker_path = config.STACKER_DIR / "model.txt"
    disabled = (config.STACKER_DIR / "disabled.json").exists()
    if decision.get("score_source") == "stacker" and stacker_path.exists() and not disabled:
        import claimants
        import stacker as stacker_mod
        s1 = pair_tables.load_prepared("test", 1)
        others = {"S2": pair_tables.load_prepared("test", 2), "S3": pair_tables.load_prepared("test", 3)}
        index = claimants.ClaimantIndex(s1)
        packed_tmp = decide.pack(scored.s1_id.to_numpy(), scored.cand_id.to_numpy(), probs)
        feats = claimants.pair_features(packed_tmp["s1_ids"], packed_tmp["cand_ids"], s1, others, index)
        x = stacker_mod.matrix(packed_tmp["scores"], packed_tmp["ranks_source"], packed_tmp["maxima"], feats)
        probs = stacker_mod.predict(stacker_mod.load(stacker_path), x)
        packed = decide.pack(packed_tmp["s1_ids"], packed_tmp["cand_ids"], probs)
    else:
        packed = decide.pack(scored.s1_id.to_numpy(), scored.cand_id.to_numpy(), probs)
    triples = decide.select(packed, rule["threshold"], rule["cap"], rule["alpha"], rule["gate"])
    matches = decide.unique_mapping(triples)

    s1_ids = pair_tables.load_prepared("test", 1).id.to_numpy()
    candidates: dict[str, list[str]] = {entity: [] for entity in s1_ids}
    for entity, cand_id in zip(table.s1_id.to_numpy(), table.cand_id.to_numpy()):
        candidates[entity].append(str(cand_id))

    matching_path = config.OUTPUT_DIR / "matching_results.tsv"
    candidate_path = config.OUTPUT_DIR / "candidate_pairs.tsv"
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
    write_report("05_predict_test", {
        "entities": int(len(s1_ids)),
        "candidate_pairs": int(len(table)),
        "matched_pairs": int(sum(len(v) for v in matches.values())),
        "validator_returncode": result.returncode,
        "validator_stdout": result.stdout,
        "validator_stderr": result.stderr,
        "rule": rule,
        "temperature": temperature,
    })
    distributed.cleanup()
    if result.returncode != 0:
        raise SystemExit(result.stdout + result.stderr)


if __name__ == "__main__":
    main()
