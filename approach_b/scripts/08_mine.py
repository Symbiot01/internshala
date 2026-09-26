"""Mine Indic / alias / unigram tables from training gold (hold-out excluded)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mining


def main() -> None:
    report = mining.run("train")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
