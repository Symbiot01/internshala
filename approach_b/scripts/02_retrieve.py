"""Stage 2 and 3: candidate retrieval, one stage per call.

  --stage entities  (train only) choose training entities and both hold-out halves
  --stage lexical   TF-IDF top-50 per source
  --stage dense     e5 top-50 per source (--dense-input native|roman, --encoder <path>)
  --stage fuse      RRF of lexical and dense, recall curves, top-K cut

Everything lands in cache/<split>/ and reports/02_<stage>_<split>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import candidates
import config
import data
import distributed
import text

SOURCES = (2, 3)
KS = [1, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100]


def split_dir(split: str) -> Path:
    return config.CACHE_DIR / split


def load_prepared(split: str, source: int) -> pd.DataFrame:
    return pd.read_parquet(split_dir(split) / f"source{source}.parquet")


def query_positions(split: str) -> np.ndarray:
    if split == "test":
        return np.arange(len(load_prepared("test", 1)), dtype=np.int64)
    groups = np.load(split_dir("train") / "entities.npz")
    return np.concatenate([groups["train"], groups["calibration"], groups["evaluation"]])


def retrieval_text(frame: pd.DataFrame, roman: bool) -> list[str]:
    if roman and "name_norm" in frame.columns:
        name = frame.name_norm.fillna(frame.name_roman)
        address = frame.address_norm.fillna(frame.address_roman)
    else:
        name = frame.name_roman if roman else frame.name
        address = frame.address_roman if roman else frame.address
    return (name.fillna("") + ", " + address.fillna("")).str.strip(", ").tolist()


def write_report(name: str, report: dict) -> None:
    if not distributed.is_main():
        return
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / f"{name}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def stage_entities() -> None:
    s1 = load_prepared("train", 1)
    truth = data.read_truth("train")
    groups = data.choose_entities(s1.id.to_numpy(), s1.country.to_numpy(), truth, config.TRAIN_ENTITIES)
    np.savez(split_dir("train") / "entities.npz", **groups)
    write_report("02_entities_train", {name: int(len(rows)) for name, rows in groups.items()})


def stage_lexical(split: str) -> None:
    import lexical

    s1 = load_prepared(split, 1)
    queries = retrieval_text(s1.iloc[query_positions(split)], roman=True)
    report = {}
    for source in SOURCES:
        start = time.time()
        index = lexical.build_index(retrieval_text(load_prepared(split, source), roman=True))
        positions, scores = lexical.search(index, queries, config.LEXICAL_TOP_K)
        np.savez(split_dir(split) / f"lexical_s{source}.npz", positions=positions, scores=scores)
        report[f"source{source}_seconds"] = round(time.time() - start, 1)
        print(source, report, flush=True)
    write_report(f"02_lexical_{split}", report)


def stage_dense(split: str, mode: str, encoder_path: str, rank: int, world: int, device) -> None:
    import dense

    tokenizer, model = dense.load_encoder(encoder_path, device=device)
    tag = f"{mode}_{Path(encoder_path).name}"
    s1 = load_prepared(split, 1)
    gather = config.CACHE_DIR / split / "tmp_dense"
    start = time.time()
    query_vectors = dense.encode(
        retrieval_text(s1.iloc[query_positions(split)], roman=(mode == "roman")),
        tokenizer, model, device=device, rank=rank, world=world, gather_dir=gather / "query",
    )
    report = {"query_seconds": round(time.time() - start, 1)}
    for source in SOURCES:
        start = time.time()
        doc_vectors = dense.encode(
            retrieval_text(load_prepared(split, source), roman=(mode == "roman")),
            tokenizer, model, device=device, rank=rank, world=world, gather_dir=gather / f"docs{source}",
        )
        positions, scores = dense.search(
            query_vectors, doc_vectors, config.DENSE_TOP_K,
            device=device, rank=rank, world=world, gather_dir=gather / f"search{source}",
        )
        if rank == 0:
            np.savez(split_dir(split) / f"dense_s{source}_{tag}.npz", positions=positions, scores=scores)
        report[f"source{source}_seconds"] = round(time.time() - start, 1)
        print(source, report, flush=True)
        distributed.barrier()
    write_report(f"02_dense_{split}_{tag}", report)


def gold_positions(split: str, source: int, rows: np.ndarray) -> list[np.ndarray]:
    """Row positions of each query entity's true matches inside one source."""
    s1_ids = load_prepared(split, 1).id.to_numpy()[rows]
    other = load_prepared(split, source)
    lookup = dict(zip(other.id.to_numpy(), range(len(other))))
    truth = data.read_truth(split)
    prefix = f"S{source}-"
    return [np.array([lookup[i] for i in truth.get(entity, []) if i.startswith(prefix) and i in lookup], dtype=np.int64)
            for entity in s1_ids]


def stage_fuse(split: str, dense_tag: str | None, keep: int | None) -> None:
    rows = query_positions(split)
    report: dict = {"dense_tag": dense_tag}
    fused_by_source = {}
    for source in SOURCES:
        lexical_positions = np.load(split_dir(split) / f"lexical_s{source}.npz")["positions"]
        lists = [lexical_positions]
        if dense_tag:
            lists.append(np.load(split_dir(split) / f"dense_s{source}_{dense_tag}.npz")["positions"])
        fused, fused_scores = candidates.rrf_fuse(lists, top_k=100)
        extra_names = []
        for extra in ("blank", "domain", "nonce"):
            extra_path = split_dir(split) / f"{extra}_s{source}.npz"
            if extra_path.exists():
                extra_pos = np.load(extra_path)["positions"]
                lists.append(extra_pos)
                extra_names.append(extra)
        if extra_names:
            fused, fused_scores = candidates.rrf_fuse(lists, top_k=100)
        fused_by_source[source] = (fused, fused_scores)
        if split == "train":
            holdout = np.zeros(len(rows), dtype=bool)
            groups = np.load(split_dir("train") / "entities.npz")
            holdout[len(groups["train"]):] = True
            gold = gold_positions(split, source, rows[holdout])
            report[f"source{source}"] = {
                "lexical": candidates.recall_curve(lists[0][holdout], gold, KS),
                "dense": candidates.recall_curve(lists[1][holdout], gold, KS) if dense_tag else None,
                "fused": candidates.recall_curve(fused[holdout], gold, KS),
                "extra": extra_names,
            }
            report[f"source{source}_breakdown"] = breakdown(split, source, rows[holdout], fused[holdout], gold)
    if keep is None:
        if split == "test":
            train_report = config.REPORTS_DIR / "02_fuse_train.json"
            if not train_report.exists():
                raise SystemExit("pass --keep, or run train fuse first so keep_per_source is known")
            keep = int(json.loads(train_report.read_text())["keep_per_source"])
        else:
            keep = choose_keep(report)
    report["keep_per_source"] = keep
    for source, (fused, fused_scores) in fused_by_source.items():
        np.savez(split_dir(split) / f"candidates_s{source}.npz", rows=rows,
                 positions=fused[:, :keep], scores=fused_scores[:, :keep])
    write_report(f"02_fuse_{split}", report)


def choose_keep(report: dict) -> int:
    """Smallest K whose fused recall reaches 99% of fused recall at 100, for both sources."""
    chosen = 0
    for source in SOURCES:
        curve = report[f"source{source}"]["fused"]
        target = 0.99 * curve[100]
        chosen = max(chosen, next(k for k in KS if curve[k] >= target))
    return min(chosen, config.KEEP_PER_SOURCE_MAX)


def breakdown(split: str, source: int, rows: np.ndarray, fused: np.ndarray, gold: list[np.ndarray]) -> dict:
    """Recall at 10 and 20 by Source 1 country and by script of the true match."""
    s1 = load_prepared(split, 1)
    countries = s1.country.to_numpy()[rows]
    other_names = load_prepared(split, source).name.to_numpy()
    out = {}
    for label, member in [("country=" + c, countries == c) for c in np.unique(countries)] + [
        ("script=non_latin", np.array([any(text.has_non_latin_letters(other_names[p]) for p in g) for g in gold])),
    ]:
        subset = [g for g, m in zip(gold, member) if m]
        out[label] = candidates.recall_curve(fused[member], subset, [10, 20, 50])
    return out


def _remap_search(local_positions: np.ndarray, global_index: np.ndarray) -> np.ndarray:
    out = np.full_like(local_positions, -1)
    valid = local_positions >= 0
    out[valid] = global_index[local_positions[valid]]
    return out


def stage_extra(split: str) -> None:
    """Routed sub-population indexes: blank-address names, domains, nonce addresses."""
    import lexical
    import postings
    import text as textmod

    s1 = load_prepared(split, 1)
    queries = query_positions(split)
    qframe = s1.iloc[queries]
    q_names = (qframe.name_norm if "name_norm" in qframe.columns else qframe.name_roman).fillna("").tolist()
    report: dict = {}
    for source in SOURCES:
        other = load_prepared(split, source)
        names = (other.name_norm if "name_norm" in other.columns else other.name_roman).fillna("")
        addresses = other.address.fillna("")
        blank = addresses.str.strip().eq("") & ~names.map(textmod.has_non_latin_letters)
        domain = (other.domain_like == "1") if "domain_like" in other.columns else names.map(textmod.is_domain_name)
        nonce = (other.nonce_like == "1") if "nonce_like" in other.columns else names.map(textmod.is_nonce_name)
        blank_idx = np.flatnonzero(blank.to_numpy())
        domain_idx = np.flatnonzero(np.asarray(domain))
        if len(blank_idx):
            index = lexical.build_index(names.iloc[blank_idx].tolist())
            local, _ = lexical.search(index, q_names, config.BLANK_INDEX_K)
            np.savez(split_dir(split) / f"blank_s{source}.npz", positions=_remap_search(local, blank_idx))
        if len(domain_idx):
            index = lexical.build_index(names.iloc[domain_idx].tolist())
            local, _ = lexical.search(index, q_names, config.DOMAIN_INDEX_K)
            np.savez(split_dir(split) / f"domain_s{source}.npz", positions=_remap_search(local, domain_idx))
        houses = other.house.fillna("") if "house" in other.columns else pd.Series([""] * len(other))
        streets = other.street_core.fillna("") if "street_core" in other.columns else pd.Series([""] * len(other))
        cities = other.city.fillna("") if "city" in other.columns else pd.Series([""] * len(other))
        nonce_idx = np.flatnonzero(np.asarray(nonce))
        keys = [
            f"{str(houses.iat[i]).lstrip('0')}|{streets.iat[i]}|{cities.iat[i]}"
            for i in nonce_idx
        ] if len(nonce_idx) else []
        post = postings.build(keys)
        q_keys = []
        q_house = qframe.house.fillna("") if "house" in qframe.columns else pd.Series([""] * len(qframe))
        q_street = qframe.street_core.fillna("") if "street_core" in qframe.columns else pd.Series([""] * len(qframe))
        q_city = qframe.city.fillna("") if "city" in qframe.columns else pd.Series([""] * len(qframe))
        for i in range(len(qframe)):
            q_keys.append([
                f"{str(q_house.iat[i]).lstrip('0')}|{q_street.iat[i]}|{q_city.iat[i]}",
                f"{str(q_house.iat[i]).lstrip('0')}|{q_street.iat[i]}|",
            ])
        local = postings.search_union(q_keys, post, config.NONCE_INDEX_K) if post else np.full((len(qframe), config.NONCE_INDEX_K), -1, np.int32)
        global_nonce = _remap_search(local, nonce_idx) if len(nonce_idx) else np.full((len(qframe), config.NONCE_INDEX_K), -1, np.int32)
        np.savez(split_dir(split) / f"nonce_s{source}.npz", positions=global_nonce)
        report[f"source{source}"] = {
            "blank_docs": int(len(blank_idx)),
            "domain_docs": int(len(domain_idx)),
            "nonce_docs": int(len(nonce_idx)),
            "nonce_keys": len(post),
        }
        print(source, report[f"source{source}"], flush=True)
    write_report(f"02_extra_{split}", report)


def stage_dense_ablation(split: str, rank: int, world: int, device) -> None:
    """Compare native vs already-saved roman e5 on hold-out Indic gold matches."""
    if split != "train":
        raise SystemExit("dense-ablation is defined on the train hold-out only")
    import dense

    rows = query_positions(split)
    groups = np.load(split_dir("train") / "entities.npz")
    holdout = np.zeros(len(rows), dtype=bool)
    holdout[len(groups["train"]):] = True
    tokenizer, model = dense.load_encoder(config.DENSE_MODEL, device=device)
    s1 = load_prepared(split, 1)
    gather = config.CACHE_DIR / split / "tmp_dense_ablation"
    native_query_path = split_dir(split) / "dense_query_native.npy"
    if native_query_path.exists():
        query_vectors = np.load(native_query_path)
    else:
        query_vectors = dense.encode(
            retrieval_text(s1.iloc[rows], roman=False), tokenizer, model,
            device=device, rank=rank, world=world, gather_dir=gather / "query",
        )
        if rank == 0:
            np.save(native_query_path, query_vectors)
        distributed.barrier()
        if rank != 0:
            query_vectors = np.load(native_query_path)
    report: dict = {"note": "roman lists reused from dense_s*_roman_multilingual-e5-small.npz"}
    for source in SOURCES:
        gold = gold_positions(split, source, rows[holdout])
        other_names = load_prepared(split, source).name.to_numpy()
        indic = np.array([any(text.has_non_latin_letters(other_names[p]) for p in g) for g in gold])
        gold_indic = [g for g, m in zip(gold, indic) if m]
        roman = np.load(split_dir(split) / f"dense_s{source}_roman_multilingual-e5-small.npz")["positions"]
        native_path = split_dir(split) / f"dense_s{source}_native_multilingual-e5-small.npz"
        if native_path.exists():
            native = np.load(native_path)["positions"]
        else:
            doc_vectors = dense.encode(
                retrieval_text(load_prepared(split, source), roman=False), tokenizer, model,
                device=device, rank=rank, world=world, gather_dir=gather / f"docs{source}",
            )
            native, scores = dense.search(
                query_vectors, doc_vectors, config.DENSE_TOP_K,
                device=device, rank=rank, world=world, gather_dir=gather / f"search{source}",
            )
            if rank == 0:
                np.savez(native_path, positions=native, scores=scores)
            distributed.barrier()
            if rank != 0:
                native = np.load(native_path)["positions"]
            del doc_vectors
        report[f"source{source}"] = {
            "roman_all": candidates.recall_curve(roman[holdout], gold, [10, 20, 50]),
            "native_all": candidates.recall_curve(native[holdout], gold, [10, 20, 50]),
            "roman_indic": candidates.recall_curve(roman[holdout][indic], gold_indic, [10, 20, 50]),
            "native_indic": candidates.recall_curve(native[holdout][indic], gold_indic, [10, 20, 50]),
        }
    write_report(f"02_dense_ablation_{split}", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--stage", choices=["entities", "lexical", "dense", "fuse", "dense-ablation", "extra"], required=True)
    parser.add_argument("--dense-input", choices=["native", "roman"], default="roman")
    parser.add_argument("--encoder", default=config.DENSE_MODEL)
    parser.add_argument("--dense-tag", default=None, help="which dense run to fuse, e.g. native_multilingual-e5-small")
    parser.add_argument("--keep", type=int, default=None, help="candidates per source; default from recall curve")
    args = parser.parse_args()
    if args.stage in {"dense", "dense-ablation"}:
        distributed.maybe_relaunch()
    rank, world, device = distributed.setup()
    gpu_stage = args.stage in {"dense", "dense-ablation"}
    if not gpu_stage and rank != 0:
        distributed.cleanup()
        return
    if args.stage == "entities":
        stage_entities()
    elif args.stage == "lexical":
        stage_lexical(args.split)
    elif args.stage == "dense":
        stage_dense(args.split, args.dense_input, args.encoder, rank, world, device)
    elif args.stage == "dense-ablation":
        stage_dense_ablation(args.split, rank, world, device)
    elif args.stage == "extra":
        stage_extra(args.split)
    else:
        stage_fuse(args.split, args.dense_tag, args.keep)
    distributed.cleanup()


if __name__ == "__main__":
    main()
