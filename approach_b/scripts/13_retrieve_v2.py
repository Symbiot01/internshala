"""v2 retrieval: lexical, Indic, skeleton, reverse, reserved-slot fusion.

Writes only under cache/{split}/v2/. Reuses the frozen dense npz from stage 02
until a later dense-v2 encode is available.
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
import lexical
import mining
import postings
import text as textmod

SOURCES = (2, 3)
KS = [1, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60]
RRF_KEEP = 25
RESERVED = 8
DENSE_TAG = "roman_multilingual-e5-small"


def v2_dir(split: str) -> Path:
    path = config.CACHE_DIR / split / "v2"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_base(split: str, source: int) -> pd.DataFrame:
    return pd.read_parquet(config.CACHE_DIR / split / f"source{source}.parquet")


def load_side(split: str, source: int) -> pd.DataFrame:
    return pd.read_parquet(v2_dir(split) / f"source{source}.parquet")


def query_positions(split: str) -> np.ndarray:
    if split == "test":
        return np.arange(len(load_base("test", 1)), dtype=np.int64)
    groups = np.load(config.CACHE_DIR / "train" / "entities.npz")
    return np.concatenate([groups["train"], groups["calibration"], groups["evaluation"]])


def _remap(local: np.ndarray, global_index: np.ndarray) -> np.ndarray:
    out = np.full_like(local, -1)
    valid = local >= 0
    out[valid] = global_index[local[valid]]
    return out


def reserved_fuse(main: list[np.ndarray], routed: list[np.ndarray], rrf_k: int = RRF_KEEP, reserved: int = RESERVED):
    fused, scores = candidates.rrf_fuse(main, top_k=rrf_k)
    if not routed:
        return fused, scores
    extra = np.concatenate(routed, axis=1).astype(np.int32)
    in_fused = (extra[:, :, None] == fused[:, None, :]).any(axis=2)
    extra = np.where((extra >= 0) & ~in_fused, extra, -1)
    for col in range(1, extra.shape[1]):
        dup = ((extra[:, col : col + 1] == extra[:, :col]).any(axis=1)) & (extra[:, col] >= 0)
        extra[dup, col] = -1
    order = np.argsort(extra < 0, axis=1, kind="stable")
    extra = np.take_along_axis(extra, order, axis=1)[:, :reserved]
    extra_sc = np.where(extra >= 0, 1.0 / (config.RRF_K + np.arange(1, extra.shape[1] + 1)), 0.0).astype(np.float32)
    return np.concatenate([fused, extra], axis=1), np.concatenate([scores, extra_sc], axis=1)


def gold_positions(split: str, source: int, rows: np.ndarray) -> list[np.ndarray]:
    s1_ids = load_base(split, 1).id.to_numpy()[rows]
    other = load_base(split, source)
    lookup = dict(zip(other.id.to_numpy(), range(len(other))))
    truth = data.read_truth(split)
    prefix = f"S{source}-"
    return [
        np.array([lookup[i] for i in truth.get(entity, []) if i.startswith(prefix) and i in lookup], dtype=np.int64)
        for entity in s1_ids
    ]


def stage_lexical(split: str) -> dict:
    s1_side = load_side(split, 1)
    rows = query_positions(split)
    queries = s1_side.retrieval_v2.iloc[rows].fillna("").tolist()
    report = {}
    for source in SOURCES:
        start = time.time()
        docs = load_side(split, source).retrieval_v2.fillna("").tolist()
        index = lexical.build_index(docs)
        positions, scores = lexical.search(index, queries, config.LEXICAL_TOP_K)
        np.savez(v2_dir(split) / f"lexical_s{source}.npz", positions=positions, scores=scores)
        report[f"source{source}_seconds"] = round(time.time() - start, 1)
        print("lexical", source, report[f"source{source}_seconds"], flush=True)
    return report


def stage_routed(split: str) -> dict:
    s1 = load_base(split, 1)
    s1_side = load_side(split, 1)
    rows = query_positions(split)
    qframe = s1.iloc[rows]
    q_names = s1_side.name_norm_v2.iloc[rows].fillna("").tolist()
    q_skel = s1_side.name_skel.iloc[rows].fillna("").tolist()
    report = {}
    for source in SOURCES:
        other = load_base(split, source)
        side = load_side(split, source)
        names = side.name_norm_v2.fillna("")
        addresses = other.address.fillna("")
        indic = side.is_indic.eq("1")
        blank = addresses.str.strip().eq("") & ~indic
        domain = (other.domain_like == "1") if "domain_like" in other.columns else names.map(textmod.is_domain_name)
        nonce = (other.nonce_like == "1") if "nonce_like" in other.columns else names.map(textmod.is_nonce_name)
        blank_idx = np.flatnonzero(blank.to_numpy())
        domain_idx = np.flatnonzero(np.asarray(domain))
        indic_idx = np.flatnonzero(indic.to_numpy())
        if len(blank_idx):
            index = lexical.build_index(names.iloc[blank_idx].tolist())
            local, _ = lexical.search(index, q_names, 3)
            np.savez(v2_dir(split) / f"blank_s{source}.npz", positions=_remap(local, blank_idx))
        else:
            np.savez(v2_dir(split) / f"blank_s{source}.npz", positions=np.full((len(rows), 3), -1, np.int32))
        if len(domain_idx):
            index = lexical.build_index(names.iloc[domain_idx].tolist())
            local, _ = lexical.search(index, q_names, 3)
            np.savez(v2_dir(split) / f"domain_s{source}.npz", positions=_remap(local, domain_idx))
        else:
            np.savez(v2_dir(split) / f"domain_s{source}.npz", positions=np.full((len(rows), 3), -1, np.int32))
        if len(indic_idx):
            index = lexical.build_index(names.iloc[indic_idx].tolist())
            local, _ = lexical.search(index, q_names, 5)
            np.savez(v2_dir(split) / f"indic_s{source}.npz", positions=_remap(local, indic_idx))
        else:
            np.savez(v2_dir(split) / f"indic_s{source}.npz", positions=np.full((len(rows), 5), -1, np.int32))
        skel_post = postings.build(side.name_skel.fillna("").tolist(), max_df=5000)
        skel_pos = postings.search(q_skel, skel_post, 5)
        np.savez(v2_dir(split) / f"skel_s{source}.npz", positions=skel_pos)
        nonce_idx = np.flatnonzero(np.asarray(nonce))
        houses = other.house.fillna("") if "house" in other.columns else pd.Series([""] * len(other))
        streets = other.street_core.fillna("") if "street_core" in other.columns else pd.Series([""] * len(other))
        cities = other.city.fillna("") if "city" in other.columns else pd.Series([""] * len(other))
        keys = [f"{str(houses.iat[i]).lstrip('0')}|{streets.iat[i]}|{cities.iat[i]}" for i in nonce_idx] if len(nonce_idx) else []
        post = postings.build(keys)
        q_house = qframe.house.fillna("") if "house" in qframe.columns else pd.Series([""] * len(qframe))
        q_street = qframe.street_core.fillna("") if "street_core" in qframe.columns else pd.Series([""] * len(qframe))
        q_city = qframe.city.fillna("") if "city" in qframe.columns else pd.Series([""] * len(qframe))
        q_keys = [
            [
                f"{str(q_house.iat[i]).lstrip('0')}|{q_street.iat[i]}|{q_city.iat[i]}",
                f"{str(q_house.iat[i]).lstrip('0')}|{q_street.iat[i]}|",
            ]
            for i in range(len(qframe))
        ]
        local = postings.search_union(q_keys, post, 3) if post else np.full((len(qframe), 3), -1, np.int32)
        global_nonce = _remap(local, nonce_idx) if len(nonce_idx) else np.full((len(qframe), 3), -1, np.int32)
        np.savez(v2_dir(split) / f"nonce_s{source}.npz", positions=global_nonce)
        report[f"source{source}"] = {
            "blank_docs": int(len(blank_idx)),
            "domain_docs": int(len(domain_idx)),
            "indic_docs": int(len(indic_idx)),
            "nonce_docs": int(len(nonce_idx)),
        }
        print("routed", source, report[f"source{source}"], flush=True)
    return report


def stage_reverse(split: str) -> dict:
    s1_side = load_side(split, 1)
    s1_names = s1_side.name_norm_v2.fillna("").tolist()
    index = lexical.build_index(s1_names)
    rows = query_positions(split)
    report = {}
    for source in SOURCES:
        other = load_base(split, source)
        side = load_side(split, source)
        names = side.name_norm_v2.fillna("").tolist()
        pos, scores = lexical.search(index, names, 3)
        np.savez(v2_dir(split) / f"reverse_lookup_s{source}.npz", positions=pos, scores=scores)
        blank = other.address.fillna("").str.strip().eq("")
        blank_idx = np.flatnonzero(blank.to_numpy())
        buckets: dict[int, list[int]] = {}
        for doc in blank_idx:
            for rank in range(3):
                owner = int(pos[doc, rank])
                if owner < 0:
                    continue
                buckets.setdefault(owner, []).append(int(doc))
        reverse = np.full((len(s1_names), 3), -1, dtype=np.int32)
        for owner, docs in buckets.items():
            reverse[owner, : min(3, len(docs))] = docs[:3]
        np.savez(v2_dir(split) / f"reverse_s{source}.npz", positions=reverse[rows])
        report[f"source{source}"] = {"blank": int(len(blank_idx)), "owners": int(len(buckets))}
        print("reverse", source, report[f"source{source}"], flush=True)
    return report


def _load_pos(split: str, name: str, source: int) -> np.ndarray:
    return np.load(v2_dir(split) / f"{name}_s{source}.npz")["positions"]


def dense_path(split: str, source: int, tag: str) -> Path:
    v2p = v2_dir(split) / f"dense_s{source}_{tag}.npz"
    if v2p.exists():
        return v2p
    return config.CACHE_DIR / split / f"dense_s{source}_{tag}.npz"


def stage_fuse(split: str, dense_tag: str | None) -> dict:
    rows = query_positions(split)
    report: dict = {"dense_tag": dense_tag, "rrf_keep": RRF_KEEP, "reserved": RESERVED}
    old = json.loads((config.REPORTS_DIR / "02_fuse_train.json").read_text()) if split == "train" else {}
    for source in SOURCES:
        lexical_pos = _load_pos(split, "lexical", source)
        main = [lexical_pos]
        if dense_tag and dense_path(split, source, dense_tag).exists():
            main.append(np.load(dense_path(split, source, dense_tag))["positions"])
        routed = [
            _load_pos(split, "blank", source),
            _load_pos(split, "reverse", source),
            _load_pos(split, "indic", source),
            _load_pos(split, "skel", source),
            _load_pos(split, "domain", source),
            _load_pos(split, "nonce", source),
        ]
        fused, scores = reserved_fuse(main, routed)
        np.savez(v2_dir(split) / f"candidates_v2_s{source}.npz", rows=rows, positions=fused, scores=scores)
        avg_pairs = float((fused >= 0).sum(axis=1).mean())
        report[f"source{source}_avg_pairs"] = avg_pairs
        if split == "train":
            groups = np.load(config.CACHE_DIR / "train" / "entities.npz")
            holdout = np.zeros(len(rows), dtype=bool)
            holdout[len(groups["train"]) :] = True
            gold = gold_positions(split, source, rows[holdout])
            report[f"source{source}"] = {
                "lexical": candidates.recall_curve(lexical_pos[holdout], gold, KS),
                "fused": candidates.recall_curve(fused[holdout], gold, KS),
                "avg_pairs": avg_pairs,
            }
            report[f"source{source}_breakdown"] = breakdown(split, source, rows[holdout], fused[holdout], gold)
            old_s = old.get(f"source{source}", {}).get("fused", {})
            report[f"source{source}_vs_old"] = {str(k): old_s.get(str(k) if str(k) in old_s else k) for k in (20, 25, 50)}
        print("fuse", source, report.get(f"source{source}"), flush=True)
    if split == "train":
        buckets = report.get("source2_breakdown", {})
        indic = buckets.get("script=non_latin", {})
        latin = buckets.get("script=latin", {})
        s3 = report["source3"]["fused"]
        indic_miss = 1 - float(indic.get(25, indic.get("25", 1)))
        latin_miss = 1 - float(latin.get(25, latin.get("25", 1)))
        s3_25 = float(s3[25] if 25 in s3 else s3["25"])
        old_keep = int(old.get("keep_per_source", 25))
        old_pairs = old_keep * 2
        new_pairs = report["source2_avg_pairs"] + report["source3_avg_pairs"]
        report["gate"] = {
            "indic_miss": indic_miss,
            "latin_miss": latin_miss,
            "s3_recall_25": s3_25,
            "pair_ratio": new_pairs / max(old_pairs, 1),
            "pass": bool(
                indic_miss <= 2 * max(latin_miss, 1e-6)
                and s3_25 >= 0.988
                and new_pairs <= 1.2 * old_pairs
            ),
        }
    (config.REPORTS_DIR / f"13_retrieve_v2_{split}.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in report if "breakdown" not in k}, indent=2, default=str), flush=True)
    return report


def breakdown(split: str, source: int, rows: np.ndarray, fused: np.ndarray, gold: list[np.ndarray]) -> dict:
    s1 = load_base(split, 1)
    countries = s1.country.to_numpy()[rows]
    other_names = load_base(split, source).name.to_numpy()
    other_addr = load_base(split, source).address.fillna("").to_numpy()
    indic = np.array([any(textmod.INDIC_CHAR.search(str(other_names[p])) for p in g) for g in gold])
    latin = np.array([len(g) > 0 and not any(textmod.INDIC_CHAR.search(str(other_names[p])) for p in g) for g in gold])
    blank = np.array([any(str(other_addr[p]).strip() == "" for p in g) for g in gold])
    out = {
        "script=non_latin": candidates.recall_curve(fused[indic], [g for g, m in zip(gold, indic) if m], [10, 20, 25, 50]),
        "script=latin": candidates.recall_curve(fused[latin], [g for g, m in zip(gold, latin) if m], [10, 20, 25, 50]),
        "blank": candidates.recall_curve(fused[blank], [g for g, m in zip(gold, blank) if m], [10, 20, 25, 50]),
    }
    for country in np.unique(countries):
        member = countries == country
        out[f"country={country}"] = candidates.recall_curve(fused[member], [g for g, m in zip(gold, member) if m], [10, 20, 25])
    return out


def stage_dense(split: str) -> dict:
    import dense
    import distributed

    rank, world, device = distributed.setup()
    tokenizer, model = dense.load_encoder(config.DENSE_MODEL, device=device)
    s1_side = load_side(split, 1)
    rows = query_positions(split)
    gather = config.CACHE_DIR / split / "v2" / "tmp_dense"
    start = time.time()
    query_vectors = dense.encode(
        s1_side.retrieval_v2.iloc[rows].fillna("").tolist(),
        tokenizer, model, device=device, rank=rank, world=world, gather_dir=gather / "query",
    )
    report = {"query_seconds": round(time.time() - start, 1)}
    for source in SOURCES:
        start = time.time()
        docs = load_side(split, source).retrieval_v2.fillna("").tolist()
        doc_vectors = dense.encode(
            docs, tokenizer, model, device=device, rank=rank, world=world, gather_dir=gather / f"docs{source}",
        )
        positions, scores = dense.search(
            query_vectors, doc_vectors, config.DENSE_TOP_K,
            device=device, rank=rank, world=world, gather_dir=gather / f"search{source}",
        )
        if rank == 0:
            np.savez(v2_dir(split) / f"dense_s{source}_{DENSE_TAG}.npz", positions=positions, scores=scores)
        report[f"source{source}_seconds"] = round(time.time() - start, 1)
        distributed.barrier()
        import shutil
        if rank == 0 and (gather / f"docs{source}").exists():
            shutil.rmtree(gather / f"docs{source}", ignore_errors=True)
            shutil.rmtree(gather / f"search{source}", ignore_errors=True)
    if rank == 0:
        (config.REPORTS_DIR / f"13_dense_v2_{split}.json").write_text(json.dumps(report, indent=2))
    return report


def maybe_dpr(indic_blank_miss_rate: float) -> dict:
    if indic_blank_miss_rate <= 0.005:
        return {"skipped": True, "reason": "indic+blank miss <= 0.5%"}
    script = Path(__file__).resolve().parents[0] / "03_train_retriever.py"
    import subprocess
    subprocess.check_call([sys.executable, str(script)])
    return {"skipped": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--stage", default="cpu", choices=("cpu", "lexical", "routed", "reverse", "fuse", "dense", "all"))
    parser.add_argument("--dense-tag", default=DENSE_TAG)
    args = parser.parse_args()
    if args.stage in ("dense", "all"):
        import distributed
        distributed.maybe_relaunch()
    if args.stage in ("cpu", "lexical", "all"):
        stage_lexical(args.split)
    if args.stage in ("cpu", "routed", "all"):
        stage_routed(args.split)
    if args.stage in ("cpu", "reverse", "all"):
        stage_reverse(args.split)
    if args.stage in ("cpu", "fuse", "all"):
        stage_fuse(args.split, args.dense_tag)
    if args.stage in ("dense", "all"):
        import distributed
        rank, world, _device = distributed.setup()
        stage_dense(args.split)
        if rank == 0:
            stage_fuse(args.split, DENSE_TAG)
        distributed.barrier()
        distributed.cleanup()


if __name__ == "__main__":
    main()
