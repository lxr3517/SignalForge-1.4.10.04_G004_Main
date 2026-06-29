from __future__ import annotations

import pandas as pd


def validate_minimum_schema(df: pd.DataFrame) -> dict:
    required = {"ds", "y", "series_id"}
    missing = sorted(list(required - set(df.columns)))
    return {
        "is_valid": len(missing) == 0,
        "missing_columns": missing,
        "row_count": len(df),
    }
