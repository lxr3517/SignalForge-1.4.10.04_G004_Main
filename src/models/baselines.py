from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.backtesting import infer_season_length, run_model_backtest
from src.models.utils import parse_horizon_to_periods


def _future_dates(last_date: pd.Timestamp, horizon: int, frequency: str) -> pd.DatetimeIndex:
    return pd.date_range(last_date, periods=horizon + 1, freq=frequency)[1:]


def _naive_predict(train_df: pd.DataFrame, horizon: int, config: dict) -> pd.DataFrame:
    rows = []
    for series_id, grp in train_df.groupby("series_id"):
        grp = grp.sort_values("ds")
        last_value = float(grp["y"].dropna().iloc[-1])
        for ds in _future_dates(grp["ds"].max(), horizon, config["frequency"]):
            rows.append({"series_id": series_id, "ds": ds, "yhat": last_value})
    return pd.DataFrame(rows)


def _historical_average_predict(train_df: pd.DataFrame, horizon: int, config: dict) -> pd.DataFrame:
    rows = []
    for series_id, grp in train_df.groupby("series_id"):
        grp = grp.sort_values("ds")
        avg_value = float(grp["y"].dropna().tail(min(30, len(grp))).mean())
        for ds in _future_dates(grp["ds"].max(), horizon, config["frequency"]):
            rows.append({"series_id": series_id, "ds": ds, "yhat": avg_value})
    return pd.DataFrame(rows)


def _seasonal_naive_predict(train_df: pd.DataFrame, horizon: int, config: dict) -> pd.DataFrame:
    rows = []
    season = infer_season_length(config["frequency"])
    for series_id, grp in train_df.groupby("series_id"):
        grp = grp.sort_values("ds").reset_index(drop=True)
        values = grp["y"].tolist()
        for step, ds in enumerate(_future_dates(grp["ds"].max(), horizon, config["frequency"]), start=1):
            idx = len(values) - season + ((step - 1) % season)
            if idx < 0 or idx >= len(values):
                idx = -1
            rows.append({"series_id": series_id, "ds": ds, "yhat": float(values[idx])})
    return pd.DataFrame(rows)


def _trend_drift_predict(train_df: pd.DataFrame, horizon: int, config: dict) -> pd.DataFrame:
    rows = []
    for series_id, grp in train_df.groupby("series_id"):
        grp = grp.sort_values("ds").reset_index(drop=True)
        values = pd.to_numeric(grp["y"], errors="coerce").dropna().astype(float)
        if values.empty:
            continue
        window = values.tail(min(max(6, len(values) // 2), len(values)))
        if len(window) >= 3:
            x = np.arange(len(window), dtype=float)
            slope, _ = np.polyfit(x, window.to_numpy(dtype=float), 1)
        else:
            slope = 0.0
        last_value = float(values.iloc[-1])
        for step, ds in enumerate(_future_dates(grp["ds"].max(), horizon, config["frequency"]), start=1):
            rows.append({"series_id": series_id, "ds": ds, "yhat": max(0.0, last_value + (slope * step))})
    return pd.DataFrame(rows)


def run_baseline_forecasts(df: pd.DataFrame, config: dict) -> dict:
    horizon = parse_horizon_to_periods(config["horizon"], config["frequency"])
    forecasts: list[list[dict]] = []
    metrics: list[dict] = []

    baseline_fns = {
        "naive": _naive_predict,
        "historical_average": _historical_average_predict,
        "trend_drift": _trend_drift_predict,
    }

    min_series_len = df.groupby("series_id").size().min()
    if min_series_len >= infer_season_length(config["frequency"]):
        baseline_fns["seasonal_naive"] = _seasonal_naive_predict

    for model_name, forecast_fn in baseline_fns.items():
        summary, _ = run_model_backtest(df, config, model_name, forecast_fn)
        final_forecast = forecast_fn(df, horizon, config)
        final_forecast["model"] = model_name
        forecasts.append(final_forecast.to_dict(orient="records"))
        metrics.append(summary)

    return {"forecasts": forecasts, "metrics": metrics}
