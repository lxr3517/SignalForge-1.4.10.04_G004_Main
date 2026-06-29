from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_regression_metrics
from src.models.utils import frequency_family, parse_horizon_to_periods
from src.pipeline.data_contracts import enforce_schema, first_series


@dataclass
class FoldResult:
    fold: int
    cutoff: pd.Timestamp
    metrics: dict
    predictions: pd.DataFrame


ForecastCallable = Callable[[pd.DataFrame, int, dict], pd.DataFrame]


DEFAULT_TUNING_TO_FOLDS = {
    "fast": 3,
    "balanced": 5,
    "aggressive": 8,
}


def infer_season_length(frequency: str) -> int:
    return {"D": 7, "W": 52, "M": 12}.get(frequency_family(frequency), 1)


def infer_min_train_size(df: pd.DataFrame, frequency: str) -> int:
    season = infer_season_length(frequency)
    n_dates = first_series(df, 'ds').nunique()
    family = frequency_family(frequency)
    if family == "D":
        return min(max(60, season * 4), max(10, n_dates - 1))
    if family == "W":
        return min(max(26, season), max(10, n_dates - 1))
    if family == "M":
        return min(max(18, season), max(10, n_dates - 1))
    return min(max(10, season * 2), max(5, n_dates - 1))


def generate_backtest_cutoffs(
    df: pd.DataFrame,
    config: dict,
    min_train_size: int | None = None,
    max_folds: int | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    df = enforce_schema(df)
    ordered_dates = pd.Index(sorted(pd.to_datetime(first_series(df, 'ds')).unique()))
    horizon = parse_horizon_to_periods(config["horizon"], config["frequency"])
    if min_train_size is None:
        min_train_size = infer_min_train_size(df, config["frequency"])
    if max_folds is None:
        max_folds = DEFAULT_TUNING_TO_FOLDS.get(config.get("tuning_depth", "balanced"), 5)

    splits: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for train_end_idx in range(min_train_size - 1, len(ordered_dates) - horizon):
        train_end = ordered_dates[train_end_idx]
        test_start = ordered_dates[train_end_idx + 1]
        test_end = ordered_dates[train_end_idx + horizon]
        splits.append((train_end, test_start, test_end))

    if not splits:
        return []

    if len(splits) <= max_folds:
        return splits

    idxs = np.linspace(0, len(splits) - 1, num=max_folds, dtype=int)
    return [splits[i] for i in idxs]


def prepare_train_test_split(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = first_series(df, 'ds')
    train = df.loc[ds <= train_end].copy()
    test = df.loc[(ds >= test_start) & (ds <= test_end)].copy()
    return train, test


def align_predictions_to_actuals(test_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    test_df = enforce_schema(test_df)
    pred_df = pred_df.copy()
    if 'ds' in pred_df.columns:
        pred_df['ds'] = pd.to_datetime(first_series(pred_df, 'ds'), errors='coerce')
    actual = test_df[['series_id', 'ds', 'y']].copy()
    merged = actual.merge(pred_df, on=["series_id", "ds"], how="left")
    return merged.sort_values(["series_id", "ds"]).reset_index(drop=True)


def evaluate_fold_predictions(test_df: pd.DataFrame, pred_df: pd.DataFrame, interval_level: float = 0.8) -> tuple[dict, pd.DataFrame]:
    merged = align_predictions_to_actuals(test_df, pred_df)
    metrics = compute_regression_metrics(
        merged["y"],
        merged["yhat"],
        y_lower=merged["yhat_lower"] if "yhat_lower" in merged.columns else None,
        y_upper=merged["yhat_upper"] if "yhat_upper" in merged.columns else None,
        interval_level=interval_level,
    )
    return metrics, merged


def summarize_fold_metrics(model_name: str, fold_results: list[FoldResult], interval_level: float = 0.8) -> dict:
    if not fold_results:
        return {
            "model": model_name,
            "mae": np.nan,
            "rmse": np.nan,
            "smape": np.nan,
            "mape": np.nan,
            "wape": np.nan,
            "bias": np.nan,
            "recent_accuracy": np.nan,
            "stability": np.nan,
            "overall_score": np.nan,
            "interval_coverage": np.nan,
            "interval_width": np.nan,
            "interval_score": np.nan,
            "n_folds": 0,
        }

    fold_df = pd.DataFrame([{"fold": fr.fold, **fr.metrics} for fr in fold_results]).sort_values("fold")
    recent_wape = float(fold_df.iloc[-1]["wape"]) if pd.notna(fold_df.iloc[-1]["wape"]) else np.nan
    stability = float(fold_df["wape"].std(ddof=0)) if len(fold_df) > 1 else 0.0

    summary = {
        "model": model_name,
        "mae": float(fold_df["mae"].mean()),
        "rmse": float(fold_df["rmse"].mean()),
        "smape": float(fold_df["smape"].mean()),
        "mape": float(fold_df["mape"].mean()),
        "wape": float(fold_df["wape"].mean()),
        "bias": float(fold_df["bias"].mean()),
        "recent_accuracy": recent_wape,
        "stability": stability,
        "overall_score": float(fold_df[["mae", "rmse", "smape", "wape"]].mean(axis=1).mean()),
        "interval_coverage": float(fold_df["interval_coverage"].mean()) if fold_df["interval_coverage"].notna().any() else np.nan,
        "interval_width": float(fold_df["interval_width"].mean()) if fold_df["interval_width"].notna().any() else np.nan,
        "interval_score": float(fold_df["interval_score"].mean()) if fold_df["interval_score"].notna().any() else np.nan,
        "n_folds": int(len(fold_df)),
        "target_interval": interval_level,
        "fold_metrics": fold_df.to_dict(orient="records"),
    }
    return summary


def run_model_backtest(
    df: pd.DataFrame,
    config: dict,
    model_name: str,
    forecast_fn: ForecastCallable,
    min_train_size: int | None = None,
) -> tuple[dict, list[dict]]:
    interval_level = 0.8
    df = enforce_schema(df)
    cutoffs = generate_backtest_cutoffs(df, config, min_train_size=min_train_size)
    fold_results: list[FoldResult] = []

    for fold_num, (train_end, test_start, test_end) in enumerate(cutoffs, start=1):
        train_df, test_df = prepare_train_test_split(df, train_end, test_start, test_end)
        horizon = test_df["ds"].nunique()
        if train_df.empty or test_df.empty or horizon <= 0:
            continue
        pred_df = forecast_fn(train_df, horizon, config)
        if pred_df is None or pred_df.empty:
            continue
        required_cols = {"series_id", "ds", "yhat"}
        if not required_cols.issubset(pred_df.columns):
            continue
        metrics, merged = evaluate_fold_predictions(test_df, pred_df, interval_level=interval_level)
        merged["model"] = model_name
        merged["fold"] = fold_num
        fold_results.append(FoldResult(fold=fold_num, cutoff=train_end, metrics=metrics, predictions=merged))

    summary = summarize_fold_metrics(model_name, fold_results, interval_level=interval_level)
    combined_predictions = pd.concat([fr.predictions for fr in fold_results], ignore_index=True) if fold_results else pd.DataFrame()
    return summary, combined_predictions.to_dict(orient="records") if not combined_predictions.empty else []
