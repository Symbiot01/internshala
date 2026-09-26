"""Train, hold-out evaluate, and write submission files.

Models are trained only on the provided TSVs. No external identity lookup.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.data.io import load_split, parse_id_list
from src.eval.metrics import macro_f05

ROOT = Path(__file__).resolve().parents[3]


def _truth_map(frame: pd.DataFrame) -> dict[str, set[str]]:
    return {
        row.source1_entity_id: set(parse_id_list(row.matched_entity_ids))
        for row in frame.itertuples(index=False)
    }


def cmd_inspect(args: argparse.Namespace) -> int:
    split = load_split(args.data_dir)
    for name, frame in split.items():
        print(f"{name}: {len(frame)} rows, columns={list(frame.columns)}")
    if "source1" in split:
        countries = sorted(split["source1"]["country"].unique())
        print(f"source1 countries: {countries}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    split = load_split(args.data_dir)
    if "ground_truth" not in split:
        print("ground truth file is missing", file=sys.stderr)
        return 1
    truth = _truth_map(split["ground_truth"])
    predictions = pd.read_csv(args.predictions, sep="\t", dtype=str, keep_default_na=False)
    predicted = {
        row.source1_entity_id: set(parse_id_list(row.matched_entity_ids))
        for row in predictions.itertuples(index=False)
    }
    scores = macro_f05(predicted, truth)
    print(f"macro_f05={scores['macro_f05']:.6f}")
    print(f"entities={int(scores['entities'])} singletons={int(scores['singletons'])}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entity resolution environment")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Print row counts for a split")
    inspect.add_argument("--data-dir", type=Path, default=ROOT / "dataset" / "train")
    inspect.set_defaults(func=cmd_inspect)

    evaluate = sub.add_parser("evaluate", help="Score predictions with macro F0.5")
    evaluate.add_argument("--data-dir", type=Path, default=ROOT / "dataset" / "train")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.set_defaults(func=cmd_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
