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
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def _build_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Lean feature set suited to linear models — calendar + short lags + rolling means."""
    cfg = config or {}
    df = dedupe_columns_keep_first(df.copy())
    feat = add_calendar_features(df)
    seasonal_lag = infer_season_length("D") if df["ds"].diff().dt.days.fillna(1).median() <= 1 else 12
    lag_candidates = sorted({1, 2, 3, 7, seasonal_lag})
    if cfg.get("enable_revenue_lag_modeling"):
        lag_candidates = sorted({1, 2, 3, 5, 7, 14, 21, seasonal_lag})
    feat = add_lag_features(feat, lags=lag_candidates)
    feat = add_rolling_features(feat, windows=[7, 14, 28])
    feat = ensure_event_columns(feat)
    feat = add_optional_regressor_features(feat, frequency=cfg.get("frequency", "D"), profile="balanced")
    feat = add_series_shape_features(feat, frequency=cfg.get("frequency", "D"))
    return feat.sort_values(["series_id", "ds"]).reset_index(drop=True)


def _make_pipeline(model_cls, **kwargs):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model_cls(**kwargs)),
    ])


def _fit_and_predict(train_df: pd.DataFrame, horizon: int, config: dict, model_cls, **model_kwargs) -> pd.DataFrame:
    feat = _build_features(train_df, config=config).dropna().reset_index(drop=True)
    if feat.empty:
        return pd.DataFrame(columns=["series_id", "ds", "yhat", "yhat_lower", "yhat_upper"])

    feature_cols = [c for c in feat.columns if c not in {"ds", "y", "series_id", "category", "partial_mode"}]
    feature_cols = [c for c in feature_cols if feat[c].dtype.kind in "biufc"]
    X = feat[feature_cols].fillna(0)
    y = first_series(feat, "y")

    pipe = _make_pipeline(model_cls, **model_kwargs)
    pipe.fit(X, y)

    rows = []
    for series_id, history in train_df.groupby("series_id"):
        history = history.sort_values("ds").copy().reset_index(drop=True)
        future_dates = pd.date_range(history["ds"].max(), periods=horizon + 1, freq=config["frequency"])[1:]
        augmented = history.copy()

        for ds in future_dates:
            new_row = {
                "series_id": series_id,
                "category": history["category"].iloc[-1] if "category" in history.columns else series_id,
                "ds": ds, "y": np.nan,
                "outage_flag": 0, "promo_flag": 0, "holiday_flag": 0,
            }
            augmented = pd.concat([augmented, pd.DataFrame([new_row])], ignore_index=True)
            feat_aug = _build_features(augmented, config=config)
            pred_row = feat_aug.iloc[[-1]].reindex(columns=feature_cols, fill_value=0).fillna(0)
            yhat = float(pipe.predict(pred_row)[0])
            # clamp to non-negative (revenue can't be negative)
            yhat = max(0.0, yhat)
            augmented.loc[augmented.index[-1], "y"] = yhat
            # linear model residual std as simple interval proxy
            rows.append({"series_id": series_id, "ds": ds, "yhat": yhat,
                         "yhat_lower": yhat * 0.88, "yhat_upper": yhat * 1.12})

    return pd.DataFrame(rows)


def _run_linear(train_df: pd.DataFrame, horizon: int, config: dict, model_cls, **kwargs) -> pd.DataFrame:
    return _fit_and_predict(train_df, horizon, config, model_cls, **kwargs)


def run_linear_forecast(df: pd.DataFrame, config: dict) -> dict:
    if not SKLEARN_AVAILABLE:
        return {"forecasts": [], "metrics": []}

    horizon = parse_horizon_to_periods(config["horizon"], config["frequency"])

    model_defs = {
        "ridge": (Ridge, {"alpha": 1.0}),
        "elasticnet": (ElasticNet, {"alpha": 0.25, "l1_ratio": 0.5, "max_iter": 25000, "tol": 1e-3, "random_state": 42}),
    }

    forecasts: list[list[dict]] = []
    metrics: list[dict] = []

    for model_name, (model_cls, model_kwargs) in model_defs.items():
        def _runner(train_df: pd.DataFrame, h: int, cfg: dict,
                    _cls=model_cls, _kw=model_kwargs) -> pd.DataFrame:
            return _run_linear(train_df, h, cfg, _cls, **_kw)

        summary, _ = run_model_backtest(df, config, model_name, _runner)
        final_forecast = _fit_and_predict(df, horizon, config, model_cls, **model_kwargs)
        final_forecast["model"] = model_name
        forecasts.append(final_forecast.to_dict(orient="records"))
        metrics.append(summary)

    return {"forecasts": forecasts, "metrics": metrics}
