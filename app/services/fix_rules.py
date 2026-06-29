from __future__ import annotations

from typing import Iterable
import pandas as pd

NULL_TOKENS = {'', ' ', 'n/a', 'na', 'null', 'none', '-', '--', 'nan'}


def fix_trim_headers(df: pd.DataFrame):
    before = list(df.columns)
    after = [str(c).strip() for c in before]
    out = df.copy()
    out.columns = after
    changed = sum(1 for b, a in zip(before, after) if str(b) != str(a))
    return out, {'action': 'trim_headers', 'columns_changed': changed}


def fix_standardize_null_tokens(df: pd.DataFrame):
    out = df.copy()
    replacements = 0
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            ser = out[col].astype(str)
            mask = ser.str.strip().str.lower().isin(NULL_TOKENS)
            replacements += int(mask.sum())
            out.loc[mask, col] = pd.NA
    return out, {'action': 'standardize_null_tokens', 'values_replaced': replacements}


def fix_drop_blank_rows(df: pd.DataFrame):
    out = df.copy()
    before = len(out)
    out = out.dropna(how='all')
    return out, {'action': 'drop_blank_rows', 'rows_removed': int(before - len(out))}


def fix_remove_exact_duplicates(df: pd.DataFrame):
    out = df.copy()
    before = len(out)
    out = out.drop_duplicates()
    return out, {'action': 'remove_exact_duplicates', 'rows_removed': int(before - len(out))}


def fix_sort_dates(df: pd.DataFrame, date_col: str | None):
    if not date_col or date_col not in df.columns:
        return df, {'action': 'sort_dates', 'applied': False, 'rows_affected': 0}
    out = df.copy()
    parsed = pd.to_datetime(out[date_col], errors='coerce')
    if parsed.notna().sum() == 0:
        return df, {'action': 'sort_dates', 'applied': False, 'rows_affected': 0}
    out[date_col] = parsed
    out = out.sort_values(date_col).reset_index(drop=True)
    return out, {'action': 'sort_dates', 'applied': True, 'rows_affected': int(len(out))}


def fix_remove_future_dates(df: pd.DataFrame, date_col: str | None):
    if not date_col or date_col not in df.columns:
        return df, {'action': 'remove_future_dates', 'rows_removed': 0}
    out = df.copy()
    parsed = pd.to_datetime(out[date_col], errors='coerce')
    if parsed.notna().sum() == 0:
        return df, {'action': 'remove_future_dates', 'rows_removed': 0}
    today = pd.Timestamp.today().normalize()
    keep_mask = parsed.isna() | (parsed <= today)
    removed = int((~keep_mask).sum())
    out = out.loc[keep_mask].copy()
    out[date_col] = parsed.loc[keep_mask]
    return out, {'action': 'remove_future_dates', 'rows_removed': removed}


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return pd.to_numeric(series, errors='coerce')
    ser = series.astype(str).str.strip()
    ser = ser.str.replace(r'[\$,]', '', regex=True)
    ser = ser.str.replace(r'\(([^\)]+)\)', r'-', regex=True)
    ser = ser.replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA, 'NaN': pd.NA})
    return pd.to_numeric(ser, errors='coerce')


def fix_parse_numeric_columns(df: pd.DataFrame, columns: Iterable[str]):
    out = df.copy()
    cols = [c for c in columns if c and c in out.columns]
    parsed_cols = []
    for col in cols:
        before_nonnull = int(out[col].notna().sum())
        converted = _clean_numeric_series(out[col])
        after_nonnull = int(converted.notna().sum())
        if after_nonnull >= before_nonnull * 0.5 or after_nonnull > 0:
            out[col] = converted
            parsed_cols.append({'column': col, 'non_null_after': after_nonnull})
    return out, {'action': 'parse_numeric_columns', 'columns': parsed_cols, 'columns_count': len(parsed_cols)}


def fix_trim_object_values(df: pd.DataFrame, columns: Iterable[str] | None = None):
    out = df.copy()
    cols = list(columns) if columns is not None else list(out.columns)
    affected = 0
    for col in cols:
        if col in out.columns and (pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col])):
            before = out[col].copy()
            out[col] = out[col].astype(str).str.strip().replace({'nan': pd.NA, 'None': pd.NA, 'NaT': pd.NA, '': pd.NA})
            affected += int((before.astype(str) != out[col].astype(str)).sum())
    return out, {'action': 'trim_object_values', 'values_touched': affected}
