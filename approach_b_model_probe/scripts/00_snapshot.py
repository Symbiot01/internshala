"""Copy Approach B caches and vendor libs into the probe folder. Read B only here."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config

LIB_FILES = (
    "text.py",
    "cross_encoder.py",
    "calibrate.py",
    "decide.py",
    "metrics.py",
    "data.py",
    "pairs.py",
    "distributed.py",
    "dense.py",
)

CACHE_FILES = (
    "entities.npz",
    "source1.parquet",
    "source2.parquet",
    "source3.parquet",
    "pairs_train.parquet",
    "pairs_calibration.parquet",
    "pairs_evaluation.parquet",
)
OPTIONAL_CACHE = ("matcher_logits.npz",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(src: Path, dst: Path) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return {"path": str(dst.relative_to(config.PROBE_ROOT)), "kind": "dir"}
    shutil.copy2(src, dst)
    return {"path": str(dst.relative_to(config.PROBE_ROOT)), "sha256": _sha256(dst), "bytes": dst.stat().st_size}


def main() -> None:
    b = config.B_ROOT
    if not b.is_dir():
        raise SystemExit(f"approach_b missing at {b}")
    manifest: list[dict] = []
    for name in LIB_FILES:
        src = b / name
        if not src.exists():
            raise SystemExit(f"missing {src}")
        manifest.append(_copy(src, config.LIB_DIR / name))
    train = b / "cache" / "train"
    dest_train = config.CACHE_DIR / "train"
    dest_train.mkdir(parents=True, exist_ok=True)
    for name in CACHE_FILES:
        src = train / name
        if not src.exists():
            raise SystemExit(f"missing {src}")
        manifest.append(_copy(src, dest_train / name))
    for name in OPTIONAL_CACHE:
        src = train / name
        if src.exists():
            manifest.append(_copy(src, dest_train / name))
        else:
            manifest.append({"path": name, "missing": True, "note": "run 00b_score_minilm_eval.py"})
    for source in (1, 2, 3):
        frame = pd.read_parquet(dest_train / f"source{source}.parquet")
        missing = [c for c in config.REQUIRED_PARQUET_COLUMNS if c not in frame.columns]
        if missing:
            raise SystemExit(f"source{source}.parquet missing {missing} (B prepare mid-rewrite?)")
    matcher_src = b / "models" / "matcher"
    if not matcher_src.is_dir():
        raise SystemExit(f"missing {matcher_src}")
    manifest.append(_copy(matcher_src, config.MODELS_DIR / "minilm"))
    decision = b / "reports" / "04_decision.json"
    if not decision.exists():
        raise SystemExit(f"missing {decision}")
    manifest.append(_copy(decision, config.REPORTS_DIR / "04_decision.json"))
    report = {"source": str(b), "files": manifest}
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "00_snapshot.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"copied": len(manifest), "probe": str(config.PROBE_ROOT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
