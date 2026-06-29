from __future__ import annotations

import pandas as pd

from src.evaluation.backtesting import infer_season_length, run_model_backtest
from src.models.utils import parse_horizon_to_periods

try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, HistoricAverage, Theta
except Exception:
    StatsForecast = None
    AutoARIMA = AutoETS = HistoricAverage = Theta = None

# Optional extra models — imported individually so missing ones don't break the rest
try:
    from statsforecast.models import AutoTheta
except Exception:
    AutoTheta = None

try:
    from statsforecast.models import AutoCES
except Exception:
    AutoCES = None

try:
    from statsforecast.models import MSTL
except Exception:
    MSTL = None

try:
    from statsforecast.models import DynamicOptimizedTheta
except Exception:
    DynamicOptimizedTheta = None


def _run_statsforecast_model(train_df: pd.DataFrame, horizon: int, config: dict, sf_model) -> pd.DataFrame:
    sf_df = train_df[["series_id", "ds", "y"]].copy().rename(columns={"series_id": "unique_id"})
    sf = StatsForecast(models=[sf_model], freq=config["frequency"], n_jobs=1)
    preds = sf.forecast(df=sf_df, h=horizon).rename(columns={"unique_id": "series_id"})
    model_col = [c for c in preds.columns if c not in {"series_id", "ds"}][0]
    return preds[["series_id", "ds", model_col]].rename(columns={model_col: "yhat"})


def run_statsforecast_forecast(df: pd.DataFrame, config: dict) -> dict:
    horizon = parse_horizon_to_periods(config["horizon"], config["frequency"])
    if StatsForecast is None:
        return {"forecasts": [], "metrics": []}

    season_length = infer_season_length(config["frequency"])
    freq = config["frequency"]

    model_defs: dict = {
        "statsforecast_autoarima": AutoARIMA(season_length=season_length),
        "statsforecast_autoets": AutoETS(season_length=season_length),
        "statsforecast_theta": Theta(season_length=season_length),
        "statsforecast_historicaverage": HistoricAverage(),
    }

    # Add extra models only when their class is available
    if AutoTheta is not None:
        model_defs["statsforecast_autotheta"] = AutoTheta(season_length=season_length)

    if AutoCES is not None:
        model_defs["statsforecast_autoces"] = AutoCES(season_length=season_length)

    if DynamicOptimizedTheta is not None:
        model_defs["statsforecast_dot"] = DynamicOptimizedTheta(season_length=season_length)

    # MSTL needs at least two seasonal periods — skip for monthly data with short history
    if MSTL is not None and freq == "D":
        # Daily: weekly (7) + annual (365) dual seasonality
        model_defs["statsforecast_mstl"] = MSTL(season_length=[7, 365])
    elif MSTL is not None and freq == "W":
        model_defs["statsforecast_mstl"] = MSTL(season_length=[52])
    elif MSTL is not None and freq == "M":
        model_defs["statsforecast_mstl"] = MSTL(season_length=[12])

    forecasts: list[list[dict]] = []
    metrics: list[dict] = []

    for model_name, sf_model in model_defs.items():
        try:
            summary, _ = run_model_backtest(
                df,
                config,
                model_name,
                lambda train_df, h, cfg, sf_model=sf_model: _run_statsforecast_model(train_df, h, cfg, sf_model),
            )
            final_forecast = _run_statsforecast_model(df, horizon, config, sf_model)
            final_forecast["model"] = model_name
            forecasts.append(final_forecast.to_dict(orient="records"))
            metrics.append(summary)
        except Exception as exc:
            # Individual model failure shouldn't kill the whole family
            metrics.append({"model": model_name, "error": str(exc)})

    return {"forecasts": forecasts, "metrics": metrics}

