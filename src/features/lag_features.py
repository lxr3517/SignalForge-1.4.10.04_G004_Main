from __future__ import annotations

import pandas as pd


def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    if lags is None:
        lags = [1, 7, 14, 28]
    feat = df.copy()
    feat = feat.sort_values(["series_id", "ds"]).reset_index(drop=True)
    for lag in lags:
        feat[f"lag_{lag}"] = feat.groupby("series_id")["y"].shift(lag)
    return feat
