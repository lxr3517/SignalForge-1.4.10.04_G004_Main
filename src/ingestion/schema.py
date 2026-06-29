from __future__ import annotations

import pandas as pd


def infer_candidate_columns(df: pd.DataFrame) -> list[str]:
    return list(df.columns)


def apply_column_mapping(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    category_col: str | None = None,
) -> pd.DataFrame:
    mapped = df.copy()
    mapped["ds"] = pd.to_datetime(mapped[date_col])
    mapped["y"] = pd.to_numeric(mapped[target_col], errors="coerce")
    mapped["series_id"] = "company_total"
    if category_col:
        mapped["series_id"] = mapped[category_col].astype(str)
        mapped["category"] = mapped[category_col].astype(str)
    else:
        mapped["category"] = "company_total"

    normalized = mapped[["ds", "y", "series_id", "category"]].sort_values(["series_id", "ds"])
    return normalized.reset_index(drop=True)
