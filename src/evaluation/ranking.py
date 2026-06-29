from __future__ import annotations

import numpy as np
import pandas as pd


LOWER_IS_BETTER = {
    "mae", "rmse", "smape", "mape", "wape",
    "bias_abs", "recent_accuracy", "stability",
    "overall_score", "interval_score",
}


def _normalize_series(series: pd.Series, lower_is_better: bool = True) -> pd.Series:
    clean = series.astype(float)
    if clean.nunique(dropna=True) <= 1:
        return pd.Series(np.ones(len(clean)), index=clean.index)
    min_v, max_v = clean.min(), clean.max()
    scaled = (clean - min_v) / (max_v - min_v + 1e-9)
    return 1 - scaled if lower_is_better else scaled


def _score_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    scored = metrics_df.copy()
    scored["bias_abs"] = scored["bias"].abs()
    component_weights = {
        "recent_accuracy": 0.30, "wape": 0.20, "mae": 0.15,
        "rmse": 0.10, "smape": 0.10, "bias_abs": 0.05,
        "stability": 0.05, "interval_score": 0.05,
    }
    for metric, weight in component_weights.items():
        if metric not in scored.columns:
            scored[f"score_{metric}"] = 0.0
            continue
        scored[f"score_{metric}"] = (
            _normalize_series(scored[metric], lower_is_better=(metric in LOWER_IS_BETTER)) * weight
        )
    scored["final_rank_score"] = scored[
        [c for c in scored.columns if c.startswith("score_")]
    ].sum(axis=1)
    return scored.sort_values("final_rank_score", ascending=False).reset_index(drop=True)


def _sort_key(preference: str) -> tuple[str, bool]:
    mapping = {
        "mae": ("mae", True), "rmse": ("rmse", True),
        "smape": ("smape", True), "bias": ("bias_abs", True),
        "recent_accuracy": ("recent_accuracy", True),
        "balanced": ("final_rank_score", False),
    }
    return mapping.get(preference, ("final_rank_score", False))


def _build_forecast_map(all_forecasts: list[list[dict]]) -> dict[str, pd.DataFrame]:
    forecast_map: dict[str, pd.DataFrame] = {}
    for forecast_records in all_forecasts:
        if not forecast_records:
            continue
        frame = pd.DataFrame(forecast_records)
        if frame.empty or not {"model", "ds", "yhat"}.issubset(frame.columns):
            continue
        forecast_map[str(frame["model"].iloc[0])] = frame
    return forecast_map


def _weighted_blend(top_rows: pd.DataFrame, forecast_map: dict[str, pd.DataFrame]) -> list[dict]:
    if top_rows.empty:
        return []

    frames: list[pd.DataFrame] = []
    weights: dict[str, float] = {}
    for _, row in top_rows.iterrows():
        model_name = row["model"]
        frame = forecast_map.get(model_name)
        if frame is None or frame.empty:
            continue
        weight = float(row.get("final_rank_score", 0.0))
        if weight <= 0:
            weight = 1.0
        if not {"ds", "yhat"}.issubset(frame.columns):
            continue
        temp = frame[["ds", "yhat"]].copy()
        if "yhat_lower" in frame.columns:
            temp["yhat_lower"] = frame["yhat_lower"]
        if "yhat_upper" in frame.columns:
            temp["yhat_upper"] = frame["yhat_upper"]
        temp = temp.rename(columns={
            "yhat": f"yhat_{model_name}",
            "yhat_lower": f"yhat_lower_{model_name}",
            "yhat_upper": f"yhat_upper_{model_name}",
        })
        frames.append(temp)
        weights[model_name] = weight

    if not frames:
        return []

    blend_df = frames[0]
    for f in frames[1:]:
        blend_df = blend_df.merge(f, on="ds", how="inner")

    if blend_df.empty:
        blend_df = frames[0]
        for f in frames[1:]:
            blend_df = blend_df.merge(f, on="ds", how="outer")

    for model_name, w in weights.items():
        blend_df[f"weight_{model_name}"] = w

    weight_sum = sum(weights.values())
    yhat_cols = [c for c in blend_df.columns
                 if c.startswith("yhat_")
                 and not c.startswith("yhat_lower_")
                 and not c.startswith("yhat_upper_")]
    blend_df["yhat"] = sum(
        blend_df[c].fillna(0) * weights.get(c.replace("yhat_", ""), 0)
        for c in yhat_cols
    ) / max(weight_sum, 1e-9)

    lower_cols = [c for c in blend_df.columns if c.startswith("yhat_lower_")]
    upper_cols = [c for c in blend_df.columns if c.startswith("yhat_upper_")]
    if lower_cols:
        iws = max(sum(weights.get(c.replace("yhat_lower_", ""), 0) for c in lower_cols), 1e-9)
        blend_df["yhat_lower"] = sum(
            blend_df[c].fillna(blend_df["yhat"]) * weights.get(c.replace("yhat_lower_", ""), 0)
            for c in lower_cols
        ) / iws
    if upper_cols:
        uws = max(sum(weights.get(c.replace("yhat_upper_", ""), 0) for c in upper_cols), 1e-9)
        blend_df["yhat_upper"] = sum(
            blend_df[c].fillna(blend_df["yhat"]) * weights.get(c.replace("yhat_upper_", ""), 0)
            for c in upper_cols
        ) / uws

    result = blend_df[
        [c for c in ["ds", "yhat", "yhat_lower", "yhat_upper"] if c in blend_df.columns]
    ].copy()
    result["model"] = "ensemble_blend"
    return result.sort_values("ds").to_dict(orient="records")


def choose_best_and_blend(all_metrics, all_forecasts, preference):
    metrics_df = pd.DataFrame(all_metrics).copy()
    if metrics_df.empty:
        return {"best_model": {}, "best_forecast": [], "blended_forecast": [], "model_ranking": []}
    scored = _score_metrics(metrics_df)
    key, ascending = _sort_key(preference)
    if key in scored.columns:
        scored = scored.sort_values(key, ascending=ascending).reset_index(drop=True)
    best_model = scored.iloc[0].to_dict()
    forecast_map = _build_forecast_map(all_forecasts)
    best_forecast = forecast_map.get(best_model["model"], pd.DataFrame()).to_dict(orient="records")
    top_rows = scored.head(min(3, len(scored))).copy()
    if top_rows["final_rank_score"].fillna(0).sum() <= 0:
        top_rows["final_rank_score"] = 1.0
    blended_forecast = _weighted_blend(top_rows, forecast_map)
    return {
        "best_model": best_model,
        "best_forecast": best_forecast,
        "blended_forecast": blended_forecast,
        "model_ranking": scored.to_dict(orient="records"),
    }
