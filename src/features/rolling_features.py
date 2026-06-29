from __future__ import annotations

import pandas as pd


def add_rolling_features(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    if windows is None:
        windows = [7, 14, 28]
    feat = df.copy().sort_values(["series_id", "ds"]).reset_index(drop=True)
    for window in windows:
        feat[f"rolling_mean_{window}"] = (
            feat.groupby("series_id")["y"].transform(lambda s: s.shift(1).rolling(window).mean())
        )
        feat[f"rolling_std_{window}"] = (
            feat.groupby("series_id")["y"].transform(lambda s: s.shift(1).rolling(window).std())
        )
    return feat
