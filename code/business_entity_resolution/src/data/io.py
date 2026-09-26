"""Tab-separated dataset readers. Country is an open string label."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")


def read_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [column for column in SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame = frame.loc[:, list(SOURCE_COLUMNS)].copy()
    for column in SOURCE_COLUMNS:
        frame[column] = frame[column].str.strip()
    if frame["entity_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate entity_id values")
    return frame


def read_ground_truth(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [column for column in TRUTH_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame = frame.loc[:, list(TRUTH_COLUMNS)].copy()
    for column in TRUTH_COLUMNS:
        frame[column] = frame[column].str.strip()
    if frame["source1_entity_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate source1_entity_id values")
    return frame


def parse_id_list(value: str) -> list[str]:
    if not value or not str(value).strip():
        return []
    return [token.strip() for token in str(value).split(",") if token.strip()]


def load_split(split_dir: Path) -> dict[str, pd.DataFrame]:
    split_dir = Path(split_dir)
    name = split_dir.name
    sources = {
        "source1": read_source(split_dir / f"{name}_source1.tsv"),
        "source2": read_source(split_dir / f"{name}_source2.tsv"),
        "source3": read_source(split_dir / f"{name}_source3.tsv"),
    }
    truth_path = split_dir / f"{name}_ground_truth.tsv"
    if truth_path.exists():
        sources["ground_truth"] = read_ground_truth(truth_path)
    return sources
