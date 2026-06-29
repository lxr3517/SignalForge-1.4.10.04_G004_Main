from __future__ import annotations

import numpy as np


def _safe_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true_arr) & np.isfinite(pred_arr)
    return true_arr[mask], pred_arr[mask]


def compute_regression_metrics(y_true, y_pred, y_lower=None, y_upper=None, interval_level: float = 0.8) -> dict:
    y_true, y_pred = _safe_arrays(y_true, y_pred)
    eps = 1e-9

    if len(y_true) == 0:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "smape": np.nan,
            "mape": np.nan,
            "wape": np.nan,
            "bias": np.nan,
            "mean_actual": np.nan,
            "mean_forecast": np.nan,
            "n_obs": 0,
            "interval_coverage": np.nan,
            "interval_width": np.nan,
            "interval_score": np.nan,
        }

    abs_err = np.abs(y_true - y_pred)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    smape = float(np.mean(2 * abs_err / (np.abs(y_true) + np.abs(y_pred) + eps)))
    mape = float(np.mean(abs_err / (np.abs(y_true) + eps)))
    wape = float(np.sum(abs_err) / (np.sum(np.abs(y_true)) + eps))
    bias = float(np.mean(y_pred - y_true))

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
        "mape": mape,
        "wape": wape,
        "bias": bias,
        "mean_actual": float(np.mean(y_true)),
        "mean_forecast": float(np.mean(y_pred)),
        "n_obs": int(len(y_true)),
        "interval_coverage": np.nan,
        "interval_width": np.nan,
        "interval_score": np.nan,
    }

    if y_lower is not None and y_upper is not None:
        y_lower = np.asarray(y_lower, dtype=float)
        y_upper = np.asarray(y_upper, dtype=float)
        lower = y_lower[: len(y_true)]
        upper = y_upper[: len(y_true)]
        interval_mask = np.isfinite(lower) & np.isfinite(upper)
        if interval_mask.any():
            lower = lower[interval_mask]
            upper = upper[interval_mask]
            true_for_interval = y_true[interval_mask]
            coverage = ((true_for_interval >= lower) & (true_for_interval <= upper)).mean()
            width = float(np.mean(np.maximum(upper - lower, 0.0)))
            metrics["interval_coverage"] = float(coverage)
            metrics["interval_width"] = width
            metrics["interval_score"] = float(abs(coverage - interval_level) + 0.1 * width / (np.mean(np.abs(true_for_interval)) + eps))

    return metrics
