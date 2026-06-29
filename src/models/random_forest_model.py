from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.backtesting import infer_season_length, run_model_backtest
from src.features.calendar_features import add_calendar_features
from src.features.event_features import ensure_event_columns
from src.features.lag_features import add_lag_features
from src.features.rolling_features import add_rolling_features
from src.features.advanced_features import add_optional_regressor_features, add_series_shape_features
from src.models.utils import parse_horizon_to_periods
from src.pipeline.data_contracts import dedupe_columns_keep_first, first_series

try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    RandomForestRegressor = None


def _build_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = config or {}
    df = dedupe_columns_keep_first(df.copy())
    feat = add_calendar_features(df)
    seasonal_lag = infer_season_length("D") if df["ds"].diff().dt.days.fillna(1).median() <= 1 else 12
    lag_candidates = sorted({1, 2, 3, 7, 14, seasonal_lag, seasonal_lag * 2})
    if cfg.get("enable_revenue_lag_modeling"):
        lag_candidates = sorted(set(lag_candidates) | {21, 28, 42, 56})
    feat = add_lag_features(feat, lags=lag_candidates)
    feat = add_rolling_features(feat, windows=[7, 14, 28, 56] if cfg.get("enable_revenue_lag_modeling") else [7, 14, 28])
    feat = ensure_event_columns(feat)
    feat = add_optional_regressor_features(feat, frequency=cfg.get("frequency", "D"), profile="extended" if cfg.get("enable_revenue_lag_modeling") else "balanced")
    feat = add_series_shape_features(feat, frequency=cfg.get("frequency", "D"))
    candidate_numeric = [c for c in feat.columns if c not in {"ds", "series_id", "category", "partial_mode"}]
    flag_cols = {
        f"{col}_missing_flag": feat[col].isna().astype(int)
        for col in candidate_numeric
        if feat[col].dtype.kind in "biufc"
    }
    if flag_cols:
        feat = pd.concat([feat, pd.DataFrame(flag_cols, index=feat.index)], axis=1)
    return feat.sort_values(["series_id", "ds"]).reset_index(drop=True)


def _fit_and_predict(train_df: pd.DataFrame, horizon: int, config: dict) -> tuple[pd.DataFrame, dict[str, float]]:
    feat = _build_features(train_df, config=config).dropna().reset_index(drop=True)
    if feat.empty or not SKLEARN_AVAILABLE:
        return pd.DataFrame(columns=["series_id", "ds", "yhat", "yhat_lower", "yhat_upper"]), {}

    feature_cols = [c for c in feat.columns if c not in {"ds", "y", "series_id", "category", "partial_mode"}]
    feature_cols = [c for c in feature_cols if feat[c].dtype.kind in "biufc"]
    if not feature_cols:
        return pd.DataFrame(columns=["series_id", "ds", "yhat", "yhat_lower", "yhat_upper"]), {}

    X = feat[feature_cols].fillna(0)
    y = pd.to_numeric(first_series(feat, "y"), errors="coerce")
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    if X.empty:
        return pd.DataFrame(columns=["series_id", "ds", "yhat", "yhat_lower", "yhat_upper"]), {}

    model = RandomForestRegressor(
        n_estimators=int(config.get("rf_n_estimators", 400) or 400),
        max_depth=int(config.get("rf_max_depth", 14) or 14),
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)

    rows = []
    importances = dict(zip(feature_cols, getattr(model, 'feature_importances_', np.zeros(len(feature_cols))).tolist()))

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
            pred_row = feat_aug.iloc[[-1]].reindex(columns=feature_cols, fill_value=0).fillna(0)
            yhat = float(model.predict(pred_row)[0])
            yhat = max(0.0, yhat)
            augmented.loc[augmented.index[-1], "y"] = yhat
            rows.append({
                "series_id": series_id,
                "ds": ds,
                "yhat": yhat,
                "yhat_lower": yhat * 0.87,
                "yhat_upper": yhat * 1.13,
            })

    return pd.DataFrame(rows), importances


def run_random_forest_forecast(df: pd.DataFrame, config: dict) -> dict:
    if not SKLEARN_AVAILABLE:
        return {"forecasts": [], "metrics": []}

    horizon = parse_horizon_to_periods(config["horizon"], config["frequency"])

    def _runner(train_df: pd.DataFrame, h: int, cfg: dict) -> pd.DataFrame:
        out, _ = _fit_and_predict(train_df, h, cfg)
        return out

    summary, _ = run_model_backtest(df, config, "random_forest", _runner)
    forecast, importances = _fit_and_predict(df, horizon, config)
    forecast["model"] = "random_forest"
    summary["feature_importance"] = importances
    return {
        "forecasts": [forecast.to_dict(orient="records")],
        "metrics": [summary],
    }
