
from __future__ import annotations

import pandas as pd
import warnings

pd.set_option('future.no_silent_downcasting', True)


def first_series(df: pd.DataFrame, col: str) -> pd.Series:
    out = df[col]
    if isinstance(out, pd.DataFrame):
        out = out.iloc[:, 0]
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=df.index, name=col)
    out = out.copy()
    out.name = col
    return out


def dedupe_columns_keep_first(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df.copy()
    out = pd.DataFrame(index=df.index)
    for col in pd.unique(df.columns):
        same = df.loc[:, df.columns == col]
        if same.shape[1] == 1:
            out[col] = same.iloc[:, 0]
            continue
        numeric = pd.concat([pd.to_numeric(same.iloc[:, i], errors='coerce') for i in range(same.shape[1])], axis=1)
        combined_numeric = numeric.bfill(axis=1).iloc[:, 0]
        fallback = same.bfill(axis=1).iloc[:, 0]
        result = fallback.where(combined_numeric.isna(), combined_numeric)
        out[col] = result.infer_objects(copy=False)
    return out



def smart_parse_dates(series: pd.Series) -> pd.Series:
    raw = first_series(pd.DataFrame({"_": series}), '_') if not isinstance(series, pd.Series) else series.copy()

    if pd.api.types.is_datetime64_any_dtype(raw):
        return pd.to_datetime(raw, errors='coerce')

    def _sensible_ratio(parsed: pd.Series) -> float:
        sensible = parsed.dropna()
        if sensible.empty:
            return 0.0
        return float(((sensible.dt.year >= 1990) & (sensible.dt.year <= 2100)).mean())

    def _safe_numeric_parse(numeric: pd.Series, kind: str):
        try:
            if kind == 'excel_days':
                parsed = pd.to_datetime('1899-12-30') + pd.to_timedelta(numeric, unit='D')
            elif kind == 'unix_seconds':
                parsed = pd.to_datetime(numeric, unit='s', origin='unix', errors='coerce')
            elif kind == 'unix_millis':
                parsed = pd.to_datetime(numeric, unit='ms', origin='unix', errors='coerce')
            elif kind == 'unix_micros':
                parsed = pd.to_datetime(numeric, unit='us', origin='unix', errors='coerce')
            elif kind == 'unix_nanos':
                parsed = pd.to_datetime(numeric, unit='ns', origin='unix', errors='coerce')
            else:
                return None
        except Exception:
            return None
        return parsed if _sensible_ratio(parsed) >= 0.5 else None

    numeric = pd.to_numeric(raw, errors='coerce')
    if numeric.notna().any():
        parsed_candidates = []
        finite = numeric[numeric.notna()]
        abs_max = float(finite.abs().max()) if not finite.empty else 0.0

        if abs_max <= 100000:
            candidate = _safe_numeric_parse(numeric, 'excel_days')
            if candidate is not None:
                parsed_candidates.append(candidate)

        if abs_max >= 1e17:
            candidate = _safe_numeric_parse(numeric, 'unix_nanos')
            if candidate is not None:
                parsed_candidates.append(candidate)
        if abs_max >= 1e14:
            candidate = _safe_numeric_parse(numeric, 'unix_micros')
            if candidate is not None:
                parsed_candidates.append(candidate)
        if abs_max >= 1e11:
            candidate = _safe_numeric_parse(numeric, 'unix_millis')
            if candidate is not None:
                parsed_candidates.append(candidate)
        if abs_max >= 1e8:
            candidate = _safe_numeric_parse(numeric, 'unix_seconds')
            if candidate is not None:
                parsed_candidates.append(candidate)

        if parsed_candidates:
            parsed_candidates.sort(key=_sensible_ratio, reverse=True)
            return parsed_candidates[0]

    text_values = raw.astype(str).str.strip()
    known_formats = [
        '%Y-%b', '%Y-%B', '%b-%Y', '%B-%Y',
        '%Y-%m', '%Y/%m', '%Y%m',
        '%Y-%m-%d', '%Y/%m/%d',
        '%m/%d/%Y', '%m-%d-%Y',
        '%d/%m/%Y', '%d-%m-%Y',
        '%d-%b-%Y', '%d-%B-%Y',
    ]
    best = None
    best_ratio = -1.0
    for fmt in known_formats:
        try:
            parsed = pd.to_datetime(text_values, format=fmt, errors='coerce')
        except Exception:
            continue
        ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
        if ratio > best_ratio:
            best = parsed
            best_ratio = ratio
        if ratio >= 0.9:
            return parsed

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        fallback = pd.to_datetime(text_values, errors='coerce')
    fallback_ratio = float(fallback.notna().mean()) if len(fallback) else 0.0
    if fallback_ratio >= best_ratio:
        return fallback
    return best if best is not None else fallback

def normalize_date_series(series: pd.Series) -> pd.Series:
    parsed = smart_parse_dates(series)
    return pd.to_datetime(parsed, errors='coerce').dt.normalize()


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = dedupe_columns_keep_first(df)
    if 'ds' not in df.columns:
        raise ValueError('Missing required column: ds')
    if 'y' not in df.columns:
        raise ValueError('Missing required column: y')

    df['ds'] = normalize_date_series(first_series(df, 'ds'))
    df['y'] = pd.to_numeric(first_series(df, 'y'), errors='coerce')

    for col in list(df.columns):
        if col in {'ds', 'y'}:
            continue
        value = df[col]
        if isinstance(value, pd.DataFrame):
            df[col] = value.iloc[:, 0]

    df = df.dropna(subset=['ds']).sort_values('ds').reset_index(drop=True)
    return df
