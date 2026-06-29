from __future__ import annotations

import pandas as pd


def add_default_intervals(df: pd.DataFrame, pct: float = 0.08) -> pd.DataFrame:
    out = df.copy()
    out["yhat_lower"] = out["yhat"] * (1 - pct)
    out["yhat_upper"] = out["yhat"] * (1 + pct)
    return out
