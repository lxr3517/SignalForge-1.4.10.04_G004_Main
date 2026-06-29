from __future__ import annotations

import pandas as pd

from src.evaluation.backtesting import infer_season_length, run_model_backtest
from src.models.utils import parse_horizon_to_periods

try:
    from prophet import Prophet
except Exception:
    Prophet = None


def _make_prophet(train_df: pd.DataFrame, config: dict) -> Prophet:
    season_length = infer_season_length(config["frequency"])
    model = Prophet(
        interval_width=0.8,
        daily_seasonality=(config["frequency"] == "D"),
        weekly_seasonality=(config["frequency"] == "D"),
        yearly_seasonality=(season_length >= 12),
    )
    for col in ["outage_flag", "promo_flag", "holiday_flag"]:
        if col in train_df.columns:
            model.add_regressor(col)
    return model


def _run_prophet(train_df: pd.DataFrame, horizon: int, config: dict) -> pd.DataFrame:
    rows = []
    for series_id, grp in train_df.groupby("series_id"):
        grp = grp.sort_values("ds").copy()
        if len(grp) < 10:
            continue
        model = _make_prophet(grp, config)
        prophet_df = grp[["ds", "y"]].copy()
        for col in ["outage_flag", "promo_flag", "holiday_flag"]:
            if col in grp.columns:
                prophet_df[col] = grp[col].astype(float)
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=horizon, freq=config["frequency"])
        for col in ["outage_flag", "promo_flag", "holiday_flag"]:
            if col in prophet_df.columns:
                future[col] = 0.0
                future.loc[future["ds"].isin(prophet_df["ds"]), col] = prophet_df.set_index("ds")[col].reindex(future["ds"]).fillna(0).values
        fcst = model.predict(future).tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        fcst["series_id"] = series_id
        rows.append(fcst)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["series_id", "ds", "yhat", "yhat_lower", "yhat_upper"])


def run_prophet_forecast(df: pd.DataFrame, config: dict) -> dict:
    horizon = parse_horizon_to_periods(config["horizon"], config["frequency"])
    if Prophet is None:
        return {"forecasts": [], "metrics": []}

    summary, _ = run_model_backtest(df, config, "prophet", _run_prophet)
    forecast = _run_prophet(df, horizon, config)
    if not forecast.empty:
        forecast["model"] = "prophet"
        return {"forecasts": [forecast.to_dict(orient="records")], "metrics": [summary]}
    return {"forecasts": [], "metrics": [summary]}
