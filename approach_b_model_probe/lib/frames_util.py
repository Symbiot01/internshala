"""Keep MiniLM-era Ditto fields so extra P2 columns do not change serialization."""

from __future__ import annotations

import pandas as pd

KEEP = ["id", "country", "name", "address", "name_roman", "address_roman"]


def strip_prepared(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in KEEP if c in frame.columns]
    return frame.loc[:, cols]
