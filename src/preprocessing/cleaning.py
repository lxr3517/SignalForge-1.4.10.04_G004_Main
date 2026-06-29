from __future__ import annotations

import pandas as pd


def generate_quality_report(df: pd.DataFrame) -> dict:
    report = {
        "rows": int(len(df)),
        "missing_target_values": int(df["y"].isna().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "min_date": str(df["ds"].min()),
        "max_date": str(df["ds"].max()),
        "missing_dates_detected": False,
        "potential_outliers": 0,
        "structural_break_warning": False,
    }

    if len(df) > 3:
        full_range = pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")
        report["missing_dates_detected"] = len(full_range) != df["ds"].nunique()

        zscore = (df["y"] - df["y"].mean()) / (df["y"].std() if df["y"].std() else 1)
        report["potential_outliers"] = int((zscore.abs() > 3).sum())

        midpoint = len(df) // 2
        early_mean = df.iloc[:midpoint]["y"].mean()
        late_mean = df.iloc[midpoint:]["y"].mean()
        if abs(late_mean - early_mean) > max(1.0, 0.35 * abs(early_mean or 1)):
            report["structural_break_warning"] = True

    return report
