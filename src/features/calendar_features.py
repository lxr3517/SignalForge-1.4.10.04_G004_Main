from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    feat["day_of_week"] = feat["ds"].dt.dayofweek
    feat["day_of_month"] = feat["ds"].dt.day
    feat["day_of_year"] = feat["ds"].dt.dayofyear
    feat["week_of_year"] = feat["ds"].dt.isocalendar().week.astype(int)
    feat["month"] = feat["ds"].dt.month
    feat["quarter"] = feat["ds"].dt.quarter
    feat["year"] = feat["ds"].dt.year
    feat["is_weekend"] = feat["ds"].dt.dayofweek.isin([5, 6]).astype(int)
    feat["is_month_end"] = feat["ds"].dt.is_month_end.astype(int)
    feat["is_month_start"] = feat["ds"].dt.is_month_start.astype(int)
    feat["is_quarter_end"] = feat["ds"].dt.is_quarter_end.astype(int)
    feat["is_quarter_start"] = feat["ds"].dt.is_quarter_start.astype(int)

    # Cyclical encodings help tree/ML models understand seasonal wrap-around.
    feat["dow_sin"] = np.sin(2 * np.pi * feat["day_of_week"] / 7.0)
    feat["dow_cos"] = np.cos(2 * np.pi * feat["day_of_week"] / 7.0)
    feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12.0)
    feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12.0)
    feat["doy_sin"] = np.sin(2 * np.pi * feat["day_of_year"] / 365.25)
    feat["doy_cos"] = np.cos(2 * np.pi * feat["day_of_year"] / 365.25)
    return feat
