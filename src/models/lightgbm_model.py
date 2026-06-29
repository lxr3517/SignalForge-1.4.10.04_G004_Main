from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.evaluation.backtesting import infer_season_length, run_model_backtest
from src.features.calendar_features import add_calendar_features
from src.features.event_features import ensure_event_columns
from src.features.lag_features import add_lag_features
from src.features.rolling_features import add_rolling_features
from src.features.advanced_features import add_optional_regressor_features, add_series_shape_features
from src.models.utils import parse_horizon_to_periods
from src.pipeline.data_contracts import dedupe_columns_keep_first, first_series, enforce_schema


def _build_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = config or {}
    df = dedupe_columns_keep_first(df.copy())
    feat = add_calendar_features(df)
    lag_candidates = [1]
    seasonal_lag = infer_season_length("D") if df["ds"].diff().dt.days.fillna(1).median() <= 1 else 12
    for lag in [seasonal_lag, seasonal_lag * 2, 3]:
        if lag not in lag_candidates:
            lag_candidates.append(lag)

    if cfg.get("enable_revenue_lag_modeling"):
        profile = (cfg.get("revenue_lag_profile") or "standard").lower()
        profile_map = {
            "short": [2, 3, 5, 7, 14],
            "standard": [2, 3, 5, 7, 14, 21, 28, seasonal_lag],
            "long": [2, 3, 5, 7, 14, 21, 28, 42, 56, seasonal_lag, seasonal_lag * 2],
        }
        for lag in profile_map.get(profile, profile_map["standard"]):
            if lag > 0 and lag not in lag_candidates:
                lag_candidates.append(lag)

    feat = add_lag_features(feat, lags=sorted(set(lag_candidates)))

    rolling_windows = [3, 7, 14, 28]
    if cfg.get("enable_revenue_lag_modeling"):
        profile = (cfg.get("revenue_lag_profile") or "standard").lower()
        if profile == "short":
            rolling_windows = [3, 7, 14]
        elif profile == "long":
            rolling_windows = [3, 7, 14, 28, 56, 84]
    feat = add_rolling_features(feat, windows=rolling_windows)

    if cfg.get("enable_revenue_lag_modeling"):
        group = feat.groupby("series_id")["y"]
        feat["lag_weighted_recent"] = (
            group.shift(1).fillna(0) * 0.50
            + group.shift(2).fillna(0) * 0.30
            + group.shift(7).fillna(0) * 0.20
        )
        feat["lag_weighted_extended"] = (
            group.shift(1).fillna(0) * 0.35
            + group.shift(7).fillna(0) * 0.25
            + group.shift(14).fillna(0) * 0.20
            + group.shift(28).fillna(0) * 0.20
        )
        feat["trend_vs_7d_avg"] = feat["lag_1"] - group.shift(1).rolling(7).mean().reset_index(level=0, drop=True)
        feat["trend_vs_28d_avg"] = feat["lag_1"] - group.shift(1).rolling(28).mean().reset_index(level=0, drop=True)

    feat = ensure_event_columns(feat)
    profile = "extended" if cfg.get("enable_revenue_lag_modeling") else "balanced"
    feat = add_optional_regressor_features(feat, frequency=config.get("frequency", "D") if config else "D", profile=profile)
    feat = add_series_shape_features(feat, frequency=config.get("frequency", "D") if config else "D")

    # Stable missing-value flags and safe fill for tree-based models.
    # Build all flag columns at once to avoid fragmentation from repeated frame.insert calls.
    candidate_numeric = [c for c in feat.columns if c not in {"ds", "series_id", "category", "partial_mode"}]
    flag_cols = {
        f"{col}_missing_flag": feat[col].isna().astype(int)
        for col in candidate_numeric
        if feat[col].dtype.kind in "biufc"
    }
    if flag_cols:
        feat = pd.concat([feat, pd.DataFrame(flag_cols, index=feat.index)], axis=1)
    return feat.sort_values(["series_id", "ds"]).reset_index(drop=True)


def _fit_model(train_df: pd.DataFrame, config: dict | None = None) -> tuple[LGBMRegressor | None, list[str], pd.DataFrame, str]:
    feat = _build_features(train_df, config=config).dropna().reset_index(drop=True)
    if feat.empty:
        return None, [], feat, 'cpu'
    feature_cols = [c for c in feat.columns if c not in {"ds", "y", "series_id", "category", "partial_mode"}]
    feature_cols = [c for c in feature_cols if feat[c].dtype.kind in "biufc"]
    X = feat[feature_cols].copy()
    y = first_series(feat, "y")
    y.name = "y"
    if X.empty:
        return None, feature_cols, feat, 'cpu'
    estimator_count = int((config or {}).get('lgbm_n_estimators', 300) or 300)
    num_leaves = int((config or {}).get('lgbm_num_leaves', 31) or 31)
    model_kwargs = dict(
        n_estimators=estimator_count,
        learning_rate=0.05,
        num_leaves=num_leaves,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1,
    )
    requested_gpu = bool((config or {}).get('use_gpu', False))
    if requested_gpu:
        model_kwargs.update({
            'device_type': 'gpu',
            'max_bin': 255,
        })
    model = LGBMRegressor(**model_kwargs)
    actual_device = 'gpu' if requested_gpu else 'cpu'
    try:
        model.fit(X, y)
    except Exception:
        if not requested_gpu:
            raise
        fallback_kwargs = dict(model_kwargs)
        fallback_kwargs.pop('device_type', None)
        fallback_kwargs.pop('max_bin', None)
        model = LGBMRegressor(**fallback_kwargs)
        model.fit(X, y)
        actual_device = 'cpu_fallback'
    return model, feature_cols, feat, actual_device


def _recursive_predict(train_df: pd.DataFrame, horizon: int, config: dict) -> tuple[pd.DataFrame, dict[str, int], str]:
    model, feature_cols, feat, actual_device = _fit_model(train_df, config=config)
    if model is None or feat.empty:
        return pd.DataFrame(columns=["series_id", "ds", "yhat", "yhat_lower", "yhat_upper"]), {}, actual_device

    rows = []
    feature_importance = dict(zip(feature_cols, model.feature_importances_.tolist()))

    for series_id, history in train_df.groupby("series_id"):
        history = history.sort_values("ds").copy().reset_index(drop=True)
        future_dates = pd.date_range(history["ds"].max(), periods=horizon + 1, freq=config["frequency"])[1:]
        augmented = history.copy()

        for ds in future_dates:
            new_row = {
                "series_id": series_id,
                "category": history["category"].iloc[-1] if "category" in history.columns else series_id,
                "ds": ds,
                "y": np.nan,
                "outage_flag": 0,
                "promo_flag": 0,
                "holiday_flag": 0,
            }
            augmented = pd.concat([augmented, pd.DataFrame([new_row])], ignore_index=True)
            feat_aug = _build_features(augmented, config=config)
            pred_row = feat_aug.iloc[[-1]].copy()
            pred_row = pred_row.reindex(columns=feature_cols, fill_value=0).fillna(0)
            yhat = float(model.predict(pred_row)[0])
            augmented.loc[augmented.index[-1], "y"] = yhat
            rows.append({
                "series_id": series_id,
                "ds": ds,
                "yhat": yhat,
                "yhat_lower": yhat * 0.9,
                "yhat_upper": yhat * 1.1,
            })

    return pd.DataFrame(rows), feature_importance, actual_device


def run_lightgbm_forecast(df: pd.DataFrame, config: dict) -> dict:
    df = enforce_schema(df)
    horizon = parse_horizon_to_periods(config["horizon"], config["frequency"])
    feature_importance_holder: dict[str, int] = {}

    def forecast_fn(train_df: pd.DataFrame, h: int, cfg: dict) -> pd.DataFrame:
        pred_df, _, _ = _recursive_predict(train_df, h, cfg)
        return pred_df

    summary, _ = run_model_backtest(df, config, "lightgbm", forecast_fn)
    pred_df, feature_importance_holder, actual_device = _recursive_predict(df, horizon, config)
    if pred_df.empty:
        return {"forecasts": [], "metrics": [summary]}

    pred_df["model"] = "lightgbm"
    summary["feature_importance"] = feature_importance_holder
    summary["lightgbm_device"] = actual_device
    return {"forecasts": [pred_df.to_dict(orient="records")], "metrics": [summary]}
