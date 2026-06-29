from __future__ import annotations

import pandas as pd


def cap_outliers_iqr(df: pd.DataFrame, column: str = "y") -> pd.DataFrame:
    capped = df.copy()
    q1 = capped[column].quantile(0.25)
    q3 = capped[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    capped[column] = capped[column].clip(lower=lower, upper=upper)
    return capped
