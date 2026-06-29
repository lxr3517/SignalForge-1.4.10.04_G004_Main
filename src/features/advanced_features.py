from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.backtesting import infer_season_length
from src.pipeline.data_contracts import first_series, dedupe_columns_keep_first

EPS = 1e-6


def _clip_feature(series: pd.Series, upper_q: float = 0.995, lower: float | None = None) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    if lower is not None:
        s = s.clip(lower=lower)
    if s.notna().any():
        upper = s.quantile(upper_q)
        if pd.notna(upper):
            s = s.clip(upper=float(upper))
    return s


def _series_from_frame(df: pd.DataFrame, col: str) -> pd.Series:
    return first_series(df, col)


def _safe_group_op(df: pd.DataFrame, value_col: str, func) -> pd.Series:
    return df.groupby("series_id")[value_col].transform(func)


def add_optional_regressor_features(df: pd.DataFrame, frequency: str = "D", profile: str = "balanced") -> pd.DataFrame:
    feat = dedupe_columns_keep_first(df.copy()).sort_values(['series_id', 'ds']).reset_index(drop=True)
    season = infer_season_length(frequency)
    profile = (profile or "balanced").lower()

    lag_map = {
        "compact": [1, 2, 3, season],
        "balanced": [1, 2, 3, 7, 14, season],
        "extended": [1, 2, 3, 7, 14, 21, 28, season, season * 2],
    }
    window_map = {
        "compact": [3, 7],
        "balanced": [3, 7, 14, 28],
        "extended": [3, 7, 14, 28, 56, 84],
    }
    lags = sorted({lag for lag in lag_map.get(profile, lag_map["balanced"]) if lag > 0})
    windows = window_map.get(profile, window_map["balanced"])

    numeric_regressors = [c for c in ["leads", "cost", "roas"] if c in feat.columns]

    for col in numeric_regressors:
        feat[col] = _clip_feature(pd.to_numeric(feat[col], errors="coerce"), lower=0 if col in {'leads', 'cost'} else None)
        feat[f"{col}_is_missing"] = feat[col].isna().astype(int)
        feat[f"{col}_filled"] = feat[col].fillna(0)

        for lag in lags:
            feat[f"{col}_lag_{lag}"] = _safe_group_op(feat, col, lambda s, lag=lag: s.shift(lag))

        for window in windows:
            feat[f"{col}_roll_mean_{window}"] = _safe_group_op(
                feat, col, lambda s, window=window: s.shift(1).rolling(window, min_periods=max(2, min(window, 3))).mean()
            )
            feat[f"{col}_roll_std_{window}"] = _safe_group_op(
                feat, col, lambda s, window=window: s.shift(1).rolling(window, min_periods=max(2, min(window, 3))).std()
            )
            feat[f"{col}_roll_sum_{window}"] = _safe_group_op(
                feat, col, lambda s, window=window: s.shift(1).rolling(window, min_periods=max(2, min(window, 3))).sum()
            )

        feat[f"{col}_ewm_7"] = _safe_group_op(feat, col, lambda s: s.shift(1).ewm(span=7, adjust=False).mean())
        feat[f"{col}_ewm_28"] = _safe_group_op(feat, col, lambda s: s.shift(1).ewm(span=28, adjust=False).mean())
        feat[f"{col}_trend_1v7"] = feat[f"{col}_lag_1"] - feat.get(f"{col}_roll_mean_7", np.nan)
        feat[f"{col}_trend_7v28"] = feat.get(f"{col}_roll_mean_7", np.nan) - feat.get(f"{col}_roll_mean_28", np.nan)

    if "leads" in feat.columns:
        feat["leads_nonzero"] = (pd.to_numeric(_series_from_frame(feat, "leads"), errors="coerce").fillna(0) > 0).astype(int)
    if "cost" in feat.columns:
        feat["cost_nonzero"] = (pd.to_numeric(_series_from_frame(feat, "cost"), errors="coerce").fillna(0) > 0).astype(int)

    # Ratios that frequently matter in marketing/ecommerce forecasting.
    if "leads" in feat.columns:
        leads_safe = pd.to_numeric(_series_from_frame(feat, "leads"), errors="coerce").replace(0, np.nan)
        feat["rev_per_lead"] = _clip_feature(pd.to_numeric(_series_from_frame(feat, "y"), errors="coerce") / (leads_safe + EPS), lower=0)
        feat["log_leads"] = np.log1p(pd.to_numeric(_series_from_frame(feat, "leads"), errors="coerce").clip(lower=0))
    if "cost" in feat.columns:
        cost_safe = pd.to_numeric(_series_from_frame(feat, "cost"), errors="coerce").replace(0, np.nan)
        feat["margin_after_cost"] = pd.to_numeric(_series_from_frame(feat, "y"), errors="coerce") - pd.to_numeric(_series_from_frame(feat, "cost"), errors="coerce")
        feat["roas_calc"] = _clip_feature(pd.to_numeric(_series_from_frame(feat, "y"), errors="coerce") / (cost_safe + EPS), lower=0)
        feat["log_cost"] = np.log1p(pd.to_numeric(_series_from_frame(feat, "cost"), errors="coerce").clip(lower=0))
    if "leads" in feat.columns and "cost" in feat.columns:
        leads_safe = pd.to_numeric(_series_from_frame(feat, "leads"), errors="coerce").replace(0, np.nan)
        feat["cost_per_lead"] = _clip_feature(pd.to_numeric(_series_from_frame(feat, "cost"), errors="coerce") / (leads_safe + EPS), lower=0)
        feat["leads_x_cost"] = _clip_feature(pd.to_numeric(_series_from_frame(feat, "leads"), errors="coerce").fillna(0) * pd.to_numeric(_series_from_frame(feat, "cost"), errors="coerce").fillna(0), lower=0)

    # Interaction features with known event-style flags.
    for flag in ["event_flag", "outage_flag", "promo_flag", "holiday_flag"]:
        if flag in feat.columns:
            feat[flag] = pd.to_numeric(_series_from_frame(feat, flag), errors="coerce").fillna(0)
            if "cost" in feat.columns:
                feat[f"{flag}_x_cost"] = feat[flag] * pd.to_numeric(_series_from_frame(feat, "cost"), errors="coerce").fillna(0)
            if "leads" in feat.columns:
                feat[f"{flag}_x_leads"] = feat[flag] * pd.to_numeric(_series_from_frame(feat, "leads"), errors="coerce").fillna(0)
            feat[f"{flag}_x_lag1"] = feat[flag] * pd.to_numeric(_series_from_frame(feat, "lag_1") if "lag_1" in feat.columns else pd.Series(0, index=feat.index, name="lag_1"), errors="coerce").fillna(0)

    return feat


def add_series_shape_features(df: pd.DataFrame, frequency: str = "D") -> pd.DataFrame:
    feat = dedupe_columns_keep_first(df.copy()).sort_values(['series_id', 'ds']).reset_index(drop=True)
    season = infer_season_length(frequency)

    feat["days_from_series_start"] = feat.groupby("series_id").cumcount()
    feat["series_age_frac"] = feat.groupby("series_id")["days_from_series_start"].transform(
        lambda s: s / max(float(s.max()), 1.0)
    )
    feat["expanding_mean"] = feat.groupby("series_id")["y"].transform(lambda s: s.shift(1).expanding(min_periods=2).mean())
    feat["expanding_std"] = feat.groupby("series_id")["y"].transform(lambda s: s.shift(1).expanding(min_periods=2).std())
    feat["seasonal_diff_1"] = feat.groupby("series_id")["y"].transform(lambda s: s.diff(season))
    feat["growth_rate_1"] = feat.groupby("series_id")["y"].transform(lambda s: s.pct_change(1, fill_method=None).replace([np.inf, -np.inf], np.nan))
    feat["growth_rate_seasonal"] = feat.groupby("series_id")["y"].transform(lambda s: s.pct_change(season, fill_method=None).replace([np.inf, -np.inf], np.nan))
    return feat
