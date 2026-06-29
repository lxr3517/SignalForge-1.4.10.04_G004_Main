from __future__ import annotations
import pandas as pd


def run_quality_checks(df: pd.DataFrame, date_column: str, target_column: str) -> dict:
    report: dict[str, object] = {}
    temp = df.copy()
    temp[date_column] = pd.to_datetime(temp[date_column], errors='coerce')
    report['row_count'] = int(len(temp))
    report['duplicate_rows'] = int(temp.duplicated().sum())
    report['null_dates'] = int(temp[date_column].isna().sum())
    report['null_target'] = int(temp[target_column].isna().sum())

    valid = temp.dropna(subset=[date_column]).sort_values(date_column)
    if not valid.empty:
        date_index = pd.date_range(valid[date_column].min(), valid[date_column].max(), freq='D')
        missing_dates = date_index.difference(valid[date_column].dt.normalize().drop_duplicates())
        report['missing_dates'] = len(missing_dates)
        report['date_min'] = str(valid[date_column].min().date())
        report['date_max'] = str(valid[date_column].max().date())
    else:
        report['missing_dates'] = 0
        report['date_min'] = None
        report['date_max'] = None

    numeric = pd.to_numeric(temp[target_column], errors='coerce')
    q1, q3 = numeric.quantile([0.25, 0.75]) if numeric.notna().any() else (0, 0)
    iqr = q3 - q1
    if iqr and numeric.notna().any():
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        report['outlier_count'] = int(((numeric < lower) | (numeric > upper)).sum())
    else:
        report['outlier_count'] = 0
    return report
