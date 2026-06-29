from __future__ import annotations

import pandas as pd
from src.pipeline.data_contracts import enforce_schema


def resample_target(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    df = enforce_schema(df)
    frames = []
    for series_id, group in df.groupby('series_id', dropna=False):
        resampled = (
            group.set_index("ds")
            .sort_index()[["y"]]
            .resample(frequency)
            .sum()
            .reset_index()
        )
        resampled["series_id"] = series_id
        frames.append(resampled)
    result = pd.concat(frames, ignore_index=True)
    return result[["series_id", "ds", "y"]]
