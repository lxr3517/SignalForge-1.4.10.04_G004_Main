from __future__ import annotations

import pandas as pd


def ensure_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    for col in ["outage_flag", "promo_flag", "holiday_flag"]:
        if col not in feat.columns:
            feat[col] = 0
    return feat
