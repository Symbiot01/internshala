"""Sanity-check Approach B modules and any caches already on disk.

Run this on the new GPU machine before training:

    approach_b/.venv/bin/python scripts/verify.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import calibrate
import candidates
import config
import data
import decide
import metrics
import pairs
import text


def _check(name: str, ok: bool, detail: str = "") -> None:
    status = "ok" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        raise SystemExit(1)


def test_metrics() -> None:
    _check("empty-empty F", metrics.f_beta_from_sets(set(), set()) == 1.0)
    _check("empty-pred F", metrics.f_beta_from_sets({"a"}, set()) == 0.0)
    _check("half F0.5", abs(metrics.f_beta_from_sets({"a", "b"}, {"a", "c"}) - 0.5) < 1e-9)


def test_rrf() -> None:
    left = np.array([[1, 2, 3]], dtype=np.int32)
    right = np.array([[2, 9, 8]], dtype=np.int32)
    pos, _ = candidates.rrf_fuse([left, right], top_k=4)
    _check("RRF prefers overlap", pos[0, 0] == 2)


def test_decide_uses_full_truth() -> None:
    s1 = np.array(["A", "A", "B"])
    c = np.array(["x", "y", "z"])
    p = np.array([0.9, 0.2, 0.1])
    lab = np.array([1, 0, 0])
    truth = {"A": ["x", "missing"], "B": ["z"]}
    packed = decide.pack(s1, c, p, lab, truth)
    gold = dict(zip(packed["entities"].tolist(), packed["gold"].tolist()))
    _check("grid gold uses full truth", gold["A"] == 2 and gold["B"] == 1)
    assigned = decide.unique_mapping([("A", "x", 0.9), ("B", "x", 0.85)])
    _check("UMC is 1-1", assigned.get("A") == ["x"] and "x" not in assigned.get("B", []))


def test_text() -> None:
    n, a, nr, ar = text.prepare_record("ACME <NULL> Inc.", "221B Baker Street, 560001")
    _check("null stripped", "null" not in n)
    tagged = text.tag_numbers(a)
    _check("dk tags", "[NUM]" in tagged or "[ZIP]" in tagged)
    kept = text.summarize_address("alpha beta देवनागरी gamma", {"alpha": 1.0, "beta": 1.0}, keep_words=2)
    _check("OOV Indic kept", "देवनागरी" in kept)


def test_calibration() -> None:
    logits = np.array([-4.0, 4.0, -4.0, 4.0])
    labels = np.array([0.0, 1.0, 0.0, 1.0])
    temperature = calibrate.fit_temperature(logits, labels)
    probs = calibrate.probabilities(logits, temperature)
    _check("temperature finite", np.isfinite(temperature) and temperature > 0)
    _check("probs ordered", probs[1] > 0.9 > probs[0])


def test_splits_and_pairs() -> None:
    groups = np.load(config.CACHE_DIR / "train" / "entities.npz")
    train, cal, ev = set(groups["train"].tolist()), set(groups["calibration"].tolist()), set(groups["evaluation"].tolist())
    _check("splits disjoint", not (train & cal) and not (train & ev) and not (cal & ev))
    _check("train size", len(groups["train"]) == 200_000, str(len(groups["train"])))

    s1 = pairs.load_prepared("train", 1)
    truth = data.read_truth("train")
    train_ids = set(s1.id.to_numpy()[groups["train"]].tolist())
    cal_ids = set(s1.id.to_numpy()[groups["calibration"]].tolist())
    ev_ids = set(s1.id.to_numpy()[groups["evaluation"]].tolist())
    table = pairs.load_pairs("train")
    _check("train pairs only train entities", set(table.s1_id.unique()).issubset(train_ids))
    _check("no hold-out leak into train pairs", set(table.s1_id.unique()).isdisjoint(cal_ids | ev_ids))

    sample = table.sample(n=min(20_000, len(table)), random_state=0)
    gold = {entity: set(truth.get(entity, [])) for entity in sample.s1_id.unique()}
    predicted = np.array([int(cid in gold[s1_id]) for s1_id, cid in zip(sample.s1_id, sample.cand_id)])
    _check("pair labels match truth", np.array_equal(predicted, sample.label.to_numpy()))
    _check("no S1 ids as candidates", not sample.cand_id.astype(str).str.startswith("S1-").any())

    fuse = json.loads((config.REPORTS_DIR / "02_fuse_train.json").read_text())
    keep = int(fuse["keep_per_source"])
    cand = np.load(config.CACHE_DIR / "train" / "candidates_s2.npz")
    _check("candidate width == keep", cand["positions"].shape[1] == keep, f"{cand['positions'].shape} vs {keep}")
    _check("fused recall@25 above A", fuse["source2"]["fused"]["25"] > 0.9675)


def main() -> None:
    test_metrics()
    test_rrf()
    test_decide_uses_full_truth()
    test_text()
    test_calibration()
    if (config.CACHE_DIR / "train" / "entities.npz").exists() and pairs.pair_path("train").exists():
        test_splits_and_pairs()
    else:
        print("[skip] cache/pair checks — retrieve + pairs not on this machine yet")
    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()
