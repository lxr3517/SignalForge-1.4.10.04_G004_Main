from __future__ import annotations

import pandas as pd


def add_hierarchy_metadata(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    feat["top_level"] = "company_total"
    feat["bottom_level"] = feat["series_id"]
    return feat
