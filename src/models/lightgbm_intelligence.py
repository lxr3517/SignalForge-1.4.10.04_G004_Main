from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import pickle

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from sklearn.metrics import f1_score, log_loss, mean_absolute_error, mean_squared_error, ndcg_score, precision_score, recall_score, roc_auc_score

from src.models.lightgbm_model import _build_features
from src.pipeline.data_contracts import dedupe_columns_keep_first, first_series


MAX_TRAIN_ROWS = 12000
MAX_SHAP_ROWS = 800
MAX_OUTPUT_ROWS = 200
RANDOM_SEED = 42


DIMENSION_CANDIDATES = {
    "company": ["series_id", "category"],
    "platform": ["platform", "__cost_bucket", "category", "series_id"],
    "affiliate": ["affiliate_id", "affiliate", "Aff_param", "aff_param"],
    "campaign": ["campaign_id", "campaign", "campaign_name"],
    "whale_segment": ["whale_segment", "user_segment", "profile_id", "member_id", "customer_id", "user_id"],
}

EXEC_FEATURE_LABELS = {
    "leads": "lead volume",
    "cost": "spend pressure",
    "roas": "efficiency",
    "revenue_per_lead": "lead monetization",
    "lag_1": "recent revenue momentum",
    "lag_7": "weekly revenue memory",
    "lag_14": "two-week revenue memory",
    "lag_28": "monthly revenue memory",
    "rolling_mean_7": "recent weekly average",
    "rolling_mean_28": "monthly revenue trend",
    "dayofweek": "weekday pattern",
    "month": "seasonal pattern",
    "is_month_start": "month-start timing",
    "is_month_end": "month-end timing",
}


@dataclass
class IntelligenceContext:
    df: pd.DataFrame
    config: dict[str, Any]


DEFAULT_TOGGLES = {
    "enable_shap_root_cause": True,
    "enable_ranking_engine": True,
    "enable_lead_quality_scoring": True,
    "enable_whale_prediction": True,
    "enable_model_anomaly_detection": True,
    "enable_ml_diminishing_returns": True,
    "enable_scenario_simulation": True,
    "enable_affiliate_quality_score": True,
    "enable_confidence_layer": True,
    "enable_multistage_modeling": True,
}


def _enabled(config: dict, key: str) -> bool:
    return bool(config.get(key, DEFAULT_TOGGLES.get(key, True)))


def _artifact_dir(config: dict) -> Path | None:
    raw = config.get("lightgbm_artifact_dir") or config.get("artifact_dir")
    if not raw:
        return None
    path = Path(str(raw))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _persist_artifact(config: dict, name: str, model: Any, features: list[str], metadata: dict) -> dict:
    base = _artifact_dir(config)
    if base is None or model is None:
        return {}
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name.lower())
    model_path = base / f"{safe_name}.pkl"
    meta_path = base / f"{safe_name}_metadata.json"
    payload = {
        "name": safe_name,
        "feature_list": features,
        "hyperparameters": getattr(model, "get_params", lambda: {})(),
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        **metadata,
    }
    try:
        with model_path.open("wb") as fh:
            pickle.dump(model, fh)
        meta_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return {"model_path": str(model_path), "metadata_path": str(meta_path)}
    except Exception as exc:
        return {"artifact_error": str(exc)}


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame, limit: int = MAX_OUTPUT_ROWS) -> list[dict]:
    if frame is None or frame.empty:
        return []
    out = frame.head(limit).copy()
    return [{k: _json_value(v) for k, v in row.items()} for row in out.to_dict(orient="records")]


def _label_feature(name: str, analyst_mode: bool = False) -> str:
    if analyst_mode:
        return str(name)
    raw = str(name)
    lowered = raw.lower()
    for key, label in EXEC_FEATURE_LABELS.items():
        if key in lowered:
            return label
    cleaned = raw.replace("_log1p", "").replace("_missing_flag", "").replace("_", " ")
    return cleaned.title()


def _first_available(df: pd.DataFrame, names: list[str]) -> str | None:
    lowered = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        found = lowered.get(str(name).lower())
        if found:
            return found
    return None


def _dimension_columns(df: pd.DataFrame) -> dict[str, str]:
    cols: dict[str, str] = {}
    for dim, candidates in DIMENSION_CANDIDATES.items():
        col = _first_available(df, candidates)
        if col:
            cols[dim] = col
    return cols


def _sample_frame(df: pd.DataFrame, max_rows: int, sort_col: str = "ds") -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    if sort_col in df.columns:
        return df.sort_values(sort_col).tail(max_rows).copy()
    return df.sample(max_rows, random_state=42).copy()


def _numeric_features(frame: pd.DataFrame) -> list[str]:
    excluded = {"ds", "y", "series_id", "category", "partial_mode"}
    return [c for c in frame.columns if c not in excluded and frame[c].dtype.kind in "biufc"]


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    actual = pd.to_numeric(y_true, errors="coerce")
    pred = pd.Series(y_pred, index=actual.index)
    valid = actual.notna() & pred.notna()
    if valid.sum() == 0:
        return {}
    a = actual[valid].astype(float)
    p = pred[valid].astype(float)
    mae = float(mean_absolute_error(a, p))
    rmse = float(np.sqrt(mean_squared_error(a, p)))
    denom = float(np.abs(a).sum() or 1.0)
    return {
        "mae": mae,
        "rmse": rmse,
        "wape": float(np.abs(a - p).sum() / denom),
        "bias": float((p - a).mean()),
    }


def _fit_explain_model(df: pd.DataFrame, config: dict) -> tuple[LGBMRegressor | None, pd.DataFrame, list[str], str | None, dict]:
    work = _sample_frame(df, MAX_TRAIN_ROWS)
    feat = _build_features(work, config=config).reset_index(drop=True)
    features = _numeric_features(feat)
    if "y" in feat.columns:
        feat = feat.dropna(subset=["y"]).copy()
    if feat.empty or not features or feat["y"].nunique(dropna=True) < 2:
        return None, feat, features, "Not enough variation for LightGBM explainability.", {}
    feature_fill = feat[features].median(numeric_only=True).fillna(0)
    feat[features] = feat[features].replace([np.inf, -np.inf], np.nan).fillna(feature_fill).fillna(0)
    model = LGBMRegressor(
        n_estimators=min(int(config.get("lgbm_n_estimators", 300) or 300), 450),
        learning_rate=0.05,
        num_leaves=min(int(config.get("lgbm_num_leaves", 31) or 31), 63),
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model.fit(feat[features], first_series(feat, "y"))
    split = max(1, int(len(feat) * 0.8))
    holdout = feat.iloc[split:].copy() if len(feat) - split >= 3 else feat.tail(min(12, len(feat))).copy()
    metrics = _regression_metrics(first_series(holdout, "y"), model.predict(holdout[features])) if not holdout.empty else {}
    artifact = _persist_artifact(
        config,
        "lightgbm_revenue_explainer",
        model,
        features,
        {
            "target_definition": "period revenue",
            "training_window": {
                "start": _json_value(feat["ds"].min()) if "ds" in feat.columns else None,
                "end": _json_value(feat["ds"].max()) if "ds" in feat.columns else None,
                "rows": int(len(feat)),
            },
            "metrics": metrics,
        },
    )
    return model, feat, features, None, {"metrics": metrics, "artifact": artifact}


def _compute_shap_layer(df: pd.DataFrame, config: dict) -> dict:
    model, feat, features, error, fit_meta = _fit_explain_model(df, config)
    if model is None or feat.empty or not features:
        return {"available": False, "error": error or "LightGBM explainability could not be fit."}

    explain_rows = _sample_frame(feat.sort_values("ds"), MAX_SHAP_ROWS)
    X = explain_rows[features].copy()
    try:
        contrib = model.predict(X, pred_contrib=True)
    except Exception as exc:
        return {"available": False, "error": f"LightGBM contribution scoring failed: {exc}"}
    contrib_df = pd.DataFrame(contrib[:, : len(features)], columns=features, index=explain_rows.index)
    pred = model.predict(X)

    global_rows = (
        contrib_df.abs()
        .mean()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
        .rename(columns={"index": "feature", 0: "mean_abs_shap"})
    )
    total = float(global_rows["mean_abs_shap"].sum() or 1.0)
    global_rows["driver"] = global_rows["feature"].apply(_label_feature)
    global_rows["share_pct"] = (global_rows["mean_abs_shap"] / total * 100.0).round(2)
    global_rows["mean_abs_shap"] = global_rows["mean_abs_shap"].round(4)

    local_records = []
    signed = contrib_df.copy()
    for idx, row in explain_rows.tail(40).iterrows():
        values = signed.loc[idx].sort_values(key=lambda s: s.abs(), ascending=False).head(8)
        positive = [
            {"driver": _label_feature(k), "impact": round(float(v), 4)}
            for k, v in values.items()
            if float(v) > 0
        ][:4]
        negative = [
            {"driver": _label_feature(k), "impact": round(float(v), 4)}
            for k, v in values.items()
            if float(v) < 0
        ][:4]
        local_records.append({
            "date": _json_value(row.get("ds")),
            "series_id": _json_value(row.get("series_id", "company_total")),
            "category": _json_value(row.get("category", row.get("series_id", "company_total"))),
            "actual_revenue": round(float(row.get("y", 0.0) or 0.0), 2),
            "predicted_revenue": round(float(pred[list(explain_rows.index).index(idx)]), 2),
            "top_positive_drivers": positive,
            "top_negative_drivers": negative,
        })

    dim_cols = _dimension_columns(df)
    aggregation_rows = []
    explain_index = explain_rows.index
    base_for_dims = df.loc[df.index.intersection(explain_index)].copy()
    if len(base_for_dims) != len(explain_rows):
        base_for_dims = explain_rows.copy()
    abs_strength = contrib_df.abs().sum(axis=1)
    signed_strength = contrib_df.sum(axis=1)
    for dim, col in dim_cols.items():
        if col not in base_for_dims.columns and col not in explain_rows.columns:
            continue
        source = base_for_dims[col] if col in base_for_dims.columns else explain_rows[col]
        temp = pd.DataFrame({
            "dimension_type": dim,
            "dimension": source.astype(str).values,
            "shap_strength": abs_strength.values,
            "net_shap": signed_strength.values,
        })
        temp = temp[(temp["dimension"] != "") & (temp["dimension"].str.lower() != "nan")]
        if temp.empty:
            continue
        agg = temp.groupby(["dimension_type", "dimension"], as_index=False).agg(
            avg_abs_shap=("shap_strength", "mean"),
            avg_net_shap=("net_shap", "mean"),
            rows=("dimension", "size"),
        ).sort_values("avg_abs_shap", ascending=False).head(30)
        aggregation_rows.extend(_records(agg, 30))
    if "ds" in explain_rows.columns:
        date_temp = pd.DataFrame({
            "dimension_type": "forecast_date",
            "dimension": pd.to_datetime(explain_rows["ds"], errors="coerce").dt.strftime("%Y-%m-%d"),
            "shap_strength": abs_strength.values,
            "net_shap": signed_strength.values,
        }).dropna(subset=["dimension"])
        if not date_temp.empty:
            date_agg = date_temp.groupby(["dimension_type", "dimension"], as_index=False).agg(
                avg_abs_shap=("shap_strength", "mean"),
                avg_net_shap=("net_shap", "mean"),
                rows=("dimension", "size"),
            ).sort_values("dimension", ascending=False).head(40)
            aggregation_rows.extend(_records(date_agg, 40))

    top_positive = contrib_df.mean().sort_values(ascending=False).head(5)
    top_negative = contrib_df.mean().sort_values(ascending=True).head(5)
    positive_labels = [_label_feature(k) for k in top_positive.index if top_positive[k] > 0][:3]
    negative_labels = [_label_feature(k) for k in top_negative.index if top_negative[k] < 0][:3]
    summary = []
    if positive_labels:
        summary.append(f"Revenue forecast is supported mainly by {', '.join(positive_labels)}.")
    if negative_labels:
        summary.append(f"Risk increased where {', '.join(negative_labels)} weakened the modeled revenue path.")
    if not summary:
        summary.append("LightGBM found mixed drivers, with no single feature dominating the explanation layer.")

    return {
        "available": True,
        "executive_summary": summary,
        "global_summary": _records(global_rows, 20),
        "local_explanations": local_records,
        "aggregations": aggregation_rows,
        "artifact_profile": {
            "method": "lightgbm_pred_contrib",
            "rows_scored": int(len(explain_rows)),
            "feature_count": int(len(features)),
            "metrics": fit_meta.get("metrics", {}),
            **(fit_meta.get("artifact", {}) or {}),
        },
    }


def _dimension_month_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if work.empty or "ds" not in work.columns:
        return pd.DataFrame()
    work["ds"] = pd.to_datetime(first_series(work, "ds"), errors="coerce")
    work["month"] = work["ds"].dt.to_period("M").dt.to_timestamp()
    dim_cols = _dimension_columns(work)
    preferred = [d for d in ["platform", "affiliate", "campaign", "whale_segment"] if d in dim_cols]
    if not preferred:
        if "category" in work.columns:
            dim_cols["segment"] = "category"
            preferred = ["segment"]
        elif "series_id" in work.columns:
            dim_cols["segment"] = "series_id"
            preferred = ["segment"]
    rows = []
    for dim in preferred:
        col = dim_cols[dim]
        if col not in work.columns:
            continue
        group_cols = ["month", col]
        agg = {"y": "sum"}
        for src in ["leads", "cost", "roas", "revenue_per_lead"]:
            if src in work.columns:
                agg[src] = "sum" if src in {"leads", "cost"} else "mean"
        temp = work.groupby(group_cols, as_index=False).agg(agg).rename(columns={col: "dimension"})
        temp["dimension_type"] = dim
        rows.append(temp)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["dimension"] = out["dimension"].astype(str).str.strip()
    out = out[(out["dimension"] != "") & (out["dimension"].str.lower() != "nan")]
    out = out.sort_values(["dimension_type", "dimension", "month"]).reset_index(drop=True)
    out["revenue_lag_1"] = out.groupby(["dimension_type", "dimension"])["y"].shift(1)
    out["revenue_3m_avg"] = out.groupby(["dimension_type", "dimension"])["y"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    out["next_revenue"] = out.groupby(["dimension_type", "dimension"])["y"].shift(-1)
    if "cost" in out.columns:
        out["roas_calc"] = out["y"] / pd.to_numeric(out["cost"], errors="coerce").replace(0, np.nan)
    if "leads" in out.columns:
        out["rpl_calc"] = out["y"] / pd.to_numeric(out["leads"], errors="coerce").replace(0, np.nan)
    return out


def _recommendation(score: float, rank_pct: float, risk: float = 0.0) -> str:
    if rank_pct <= 0.2 and risk <= 0.55:
        return "scale"
    if rank_pct <= 0.45:
        return "test"
    if risk >= 0.75:
        return "caution"
    return "monitor"


def _compute_ranking_layer(df: pd.DataFrame, config: dict | None = None) -> dict:
    cfg = dict(config or {})
    frame = _dimension_month_frame(df)
    if frame.empty or frame["dimension"].nunique() < 3:
        return {"available": False, "error": "Not enough source dimensions for LightGBM ranking."}
    feature_cols = [c for c in ["y", "leads", "cost", "roas", "revenue_per_lead", "revenue_lag_1", "revenue_3m_avg", "roas_calc", "rpl_calc"] if c in frame.columns]
    train = frame.dropna(subset=["next_revenue"]).copy()
    train = train.replace([np.inf, -np.inf], np.nan)
    for c in feature_cols:
        train[c] = pd.to_numeric(train[c], errors="coerce")
    train = train.dropna(subset=feature_cols)
    if len(train) < 8 or train["next_revenue"].nunique(dropna=True) < 3:
        return {"available": False, "error": "Not enough future value variation for ranking."}
    def _rank_labels(s: pd.Series) -> pd.Series:
        if len(s) <= 1 or s.nunique(dropna=True) <= 1:
            return pd.Series(np.zeros(len(s), dtype=int), index=s.index)
        pct = s.rank(method="first", pct=True)
        return np.floor(pct * 5).clip(0, 4).astype(int)

    train["label"] = train.groupby("month")["next_revenue"].transform(_rank_labels)
    train["label"] = pd.to_numeric(train["label"], errors="coerce").fillna(0).astype(int)
    train = train.sort_values("month")
    groups = train.groupby("month").size().astype(int).tolist()
    try:
        ranker = LGBMRanker(
            objective="lambdarank",
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=15,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
        ranker.fit(train[feature_cols], train["label"], group=groups)
    except Exception as exc:
        return {"available": False, "error": f"LightGBM ranker could not fit: {exc}"}

    latest_idx = frame.groupby(["dimension_type", "dimension"])["month"].idxmax()
    latest = frame.loc[latest_idx].copy().replace([np.inf, -np.inf], np.nan)
    for c in feature_cols:
        latest[c] = pd.to_numeric(latest[c], errors="coerce")
    latest[feature_cols] = latest[feature_cols].fillna(train[feature_cols].median(numeric_only=True)).fillna(0)
    latest["ranking_score"] = ranker.predict(latest[feature_cols])
    latest["rank"] = latest.groupby("dimension_type")["ranking_score"].rank(ascending=False, method="first")
    latest["dimension_count"] = latest.groupby("dimension_type")["dimension"].transform("count")
    latest["rank_pct"] = latest["rank"] / latest["dimension_count"].clip(lower=1)
    volatility = frame.groupby(["dimension_type", "dimension"])["y"].std(ddof=0).rename("volatility")
    latest = latest.merge(volatility, on=["dimension_type", "dimension"], how="left")
    latest["risk_score"] = (pd.to_numeric(latest["volatility"], errors="coerce").fillna(0) / pd.to_numeric(latest["y"], errors="coerce").abs().replace(0, np.nan)).clip(0, 2).fillna(0) / 2
    latest["recommendation"] = [
        _recommendation(float(s), float(p), float(r))
        for s, p, r in zip(latest["ranking_score"], latest["rank_pct"], latest["risk_score"])
    ]
    latest["business_reason"] = latest.apply(
        lambda r: "High modeled upside with acceptable volatility." if r["recommendation"] == "scale"
        else "Promising but should be tested before larger budget moves." if r["recommendation"] == "test"
        else "Elevated volatility or weaker modeled upside; review before scaling." if r["recommendation"] == "caution"
        else "Keep watching until signal improves.",
        axis=1,
    )
    show_cols = ["dimension_type", "dimension", "rank", "ranking_score", "recommendation", "business_reason", "y"]
    for extra in ["leads", "cost", "roas_calc", "rpl_calc", "risk_score"]:
        if extra in latest.columns:
            show_cols.append(extra)
    ranked = latest.sort_values(["dimension_type", "rank"]).copy()
    ranked["ranking_score"] = pd.to_numeric(ranked["ranking_score"], errors="coerce").round(4)
    if "risk_score" in ranked.columns:
        ranked["risk_score"] = pd.to_numeric(ranked["risk_score"], errors="coerce").round(3)
    metrics = {}
    try:
        train_pred = ranker.predict(train[feature_cols])
        metrics["ndcg"] = float(ndcg_score([train["label"].to_numpy()], [train_pred]))
        top_cut = max(1, int(len(train) * 0.2))
        top_actual = set(train.sort_values("label", ascending=False).head(top_cut).index)
        top_pred = set(pd.Series(train_pred, index=train.index).sort_values(ascending=False).head(top_cut).index)
        metrics["top_bucket_capture"] = float(len(top_actual & top_pred) / max(len(top_actual), 1))
    except Exception:
        metrics = {}
    artifact = _persist_artifact(
        cfg,
        "lightgbm_ranker",
        ranker,
        feature_cols,
        {
            "target_definition": "next-period relative opportunity rank",
            "training_window": {
                "start": _json_value(train["month"].min()),
                "end": _json_value(train["month"].max()),
                "rows": int(len(train)),
            },
            "metrics": metrics,
        },
    )
    return {
        "available": True,
        "ranked_opportunities": _records(ranked[show_cols], MAX_OUTPUT_ROWS),
        "artifact_profile": {
            "objective": "lambdarank",
            "training_rows": int(len(train)),
            "feature_count": int(len(feature_cols)),
            "metrics": metrics,
            **artifact,
        },
    }


def _bucket_probability(p: float) -> str:
    if p >= 0.75:
        return "high"
    if p >= 0.45:
        return "medium"
    return "low"


def _value_tier(p_high: float, revenue_proxy: float) -> str:
    if p_high >= 0.6 or revenue_proxy >= 0.8:
        return "premium"
    if p_high >= 0.35 or revenue_proxy >= 0.5:
        return "growth"
    return "standard"


def _fit_classifier(train: pd.DataFrame, features: list[str], target: str) -> tuple[LGBMClassifier | None, dict]:
    y = pd.to_numeric(train[target], errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2:
        return None, {}
    model = LGBMClassifier(
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=15,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model.fit(train[features], y)
    metrics = {}
    try:
        pred = model.predict_proba(train[features])[:, 1]
        label_pred = (pred >= 0.5).astype(int)
        metrics = {
            "auc": float(roc_auc_score(y, pred)),
            "log_loss": float(log_loss(y, np.clip(pred, 1e-5, 1 - 1e-5))),
            "precision": float(precision_score(y, label_pred, zero_division=0)),
            "recall": float(recall_score(y, label_pred, zero_division=0)),
            "f1": float(f1_score(y, label_pred, zero_division=0)),
        }
    except Exception:
        metrics = {}
    return model, metrics


def _compute_lead_quality_layer(df: pd.DataFrame, config: dict | None = None) -> dict:
    cfg = dict(config or {})
    if "leads" not in df.columns:
        return {"available": False, "error": "Lead quality scoring needs a mapped leads field."}
    frame = _dimension_month_frame(df)
    if frame.empty or "leads" not in frame.columns or len(frame) < 8:
        return {"available": False, "error": "Not enough lead history for lead quality scoring."}
    frame = frame.replace([np.inf, -np.inf], np.nan).copy()
    frame["conversion_target"] = (pd.to_numeric(frame["next_revenue"], errors="coerce").fillna(0) > 0).astype(int)
    threshold = pd.to_numeric(frame["next_revenue"], errors="coerce").quantile(0.75)
    frame["high_value_target"] = (pd.to_numeric(frame["next_revenue"], errors="coerce").fillna(0) >= float(threshold or 0)).astype(int)
    features = [c for c in ["leads", "cost", "roas", "revenue_per_lead", "revenue_lag_1", "revenue_3m_avg", "roas_calc", "rpl_calc"] if c in frame.columns]
    for c in features:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    train = frame.dropna(subset=["next_revenue"]).copy()
    train[features] = train[features].fillna(train[features].median(numeric_only=True)).fillna(0)
    if len(train) < 8 or not features:
        return {"available": False, "error": "Not enough usable features for lead quality scoring."}
    conv_model, conv_metrics = _fit_classifier(train, features, "conversion_target")
    value_model, value_metrics = _fit_classifier(train, features, "high_value_target")
    if conv_model is None and value_model is None:
        return {"available": False, "error": "Lead outcomes do not have enough class variation yet."}
    latest_idx = frame.groupby(["dimension_type", "dimension"])["month"].idxmax()
    latest = frame.loc[latest_idx].copy()
    latest[features] = latest[features].fillna(train[features].median(numeric_only=True)).fillna(0)
    latest["conversion_probability"] = conv_model.predict_proba(latest[features])[:, 1] if conv_model is not None else np.nan
    latest["high_value_probability"] = value_model.predict_proba(latest[features])[:, 1] if value_model is not None else np.nan
    revenue_rank = pd.to_numeric(latest["y"], errors="coerce").rank(pct=True).fillna(0)
    latest["confidence_bucket"] = latest["conversion_probability"].fillna(latest["high_value_probability"]).fillna(0).apply(_bucket_probability)
    latest["value_tier"] = [
        _value_tier(float(hv if pd.notna(hv) else 0.0), float(rr))
        for hv, rr in zip(latest["high_value_probability"], revenue_rank)
    ]
    latest["lead_quality_summary"] = latest.apply(
        lambda r: f"{str(r['dimension'])} is a {r['confidence_bucket']} confidence, {r['value_tier']} value lead source.",
        axis=1,
    )
    show_cols = ["dimension_type", "dimension", "conversion_probability", "high_value_probability", "confidence_bucket", "value_tier", "lead_quality_summary"]
    scored = latest.sort_values(["confidence_bucket", "high_value_probability"], ascending=[True, False]).copy()
    for col in ["conversion_probability", "high_value_probability"]:
        scored[col] = pd.to_numeric(scored[col], errors="coerce").round(4)
    conv_artifact = _persist_artifact(cfg, "lightgbm_lead_conversion", conv_model, features, {
        "target_definition": "next-period revenue is positive",
        "training_window": {"start": _json_value(train["month"].min()), "end": _json_value(train["month"].max()), "rows": int(len(train))},
        "metrics": conv_metrics,
    })
    value_artifact = _persist_artifact(cfg, "lightgbm_lead_high_value", value_model, features, {
        "target_definition": "next-period revenue is in the top value bucket",
        "training_window": {"start": _json_value(train["month"].min()), "end": _json_value(train["month"].max()), "rows": int(len(train))},
        "metrics": value_metrics,
    })
    return {
        "available": True,
        "scores": _records(scored[show_cols], MAX_OUTPUT_ROWS),
        "artifact_profile": {
            "model": "LGBMClassifier",
            "training_rows": int(len(train)),
            "conversion_metrics": conv_metrics,
            "high_value_metrics": value_metrics,
            "artifacts": {"conversion": conv_artifact, "high_value": value_artifact},
        },
    }


def _severity_from_score(score: float) -> str:
    if score >= 3.5:
        return "critical"
    if score >= 2.5:
        return "notable"
    if score >= 1.75:
        return "minor"
    return "normal"


def _risk_label(score: float) -> str:
    if score >= 0.7:
        return "high risk"
    if score >= 0.4:
        return "moderate risk"
    return "low risk"


def _compute_residual_anomaly_layer(df: pd.DataFrame, config: dict, shap_layer: dict | None = None) -> dict:
    model, feat, features, error, fit_meta = _fit_explain_model(df, config)
    if model is None or feat.empty or not features:
        return {"available": False, "error": error or "Residual anomaly model could not fit."}
    scored = feat.sort_values("ds").tail(min(len(feat), 5000)).copy()
    pred = model.predict(scored[features])
    scored["predicted_revenue"] = pred
    scored["residual"] = pd.to_numeric(scored["y"], errors="coerce") - scored["predicted_revenue"]
    resid_std = float(scored["residual"].std(ddof=0) or 0.0)
    if resid_std <= 0:
        return {"available": False, "error": "Residuals are too flat for anomaly scoring."}
    scored["anomaly_score"] = (scored["residual"] / resid_std).abs()
    scored["severity"] = scored["anomaly_score"].apply(_severity_from_score)
    show = scored[scored["severity"] != "normal"].tail(50).copy()
    if show.empty:
        show = scored.tail(20).copy()
    show["date"] = pd.to_datetime(show["ds"]).dt.strftime("%Y-%m-%d")
    show["actual_revenue"] = pd.to_numeric(show["y"], errors="coerce").round(2)
    show["predicted_revenue"] = pd.to_numeric(show["predicted_revenue"], errors="coerce").round(2)
    show["anomaly_score"] = pd.to_numeric(show["anomaly_score"], errors="coerce").round(3)
    local_by_date = {}
    if shap_layer and shap_layer.get("available"):
        for row in shap_layer.get("local_explanations") or []:
            local_by_date[str(row.get("date"))] = row
    likely_drivers = []
    actions = []
    for _, row in show.iterrows():
        date_key = str(row.get("date"))
        local = local_by_date.get(date_key, {})
        drivers = []
        if float(row.get("residual", 0.0) or 0.0) < 0:
            drivers = [d.get("driver") for d in (local.get("top_negative_drivers") or []) if d.get("driver")]
            action = "Investigate revenue softness versus model expectation before scaling spend."
        else:
            drivers = [d.get("driver") for d in (local.get("top_positive_drivers") or []) if d.get("driver")]
            action = "Review whether the upside driver is repeatable before treating it as a new baseline."
        likely_drivers.append(", ".join(drivers[:3]) if drivers else "mixed model residual")
        actions.append(action)
    show["likely_drivers"] = likely_drivers
    show["recommended_action"] = actions
    return {
        "available": True,
        "anomaly_rows": _records(show[["date", "series_id", "actual_revenue", "predicted_revenue", "residual", "anomaly_score", "severity", "likely_drivers", "recommended_action"]], 50),
        "artifact_profile": {
            "method": "lightgbm_residual_expectation",
            "metrics": fit_meta.get("metrics", {}),
            **(fit_meta.get("artifact", {}) or {}),
        },
    }


def _profile_id_column(df: pd.DataFrame) -> str | None:
    return _first_available(df, ["profile_id", "member_id", "customer_id", "user_id", "account_id", "profileid", "memberid"])


def _profile_behavior_frame(df: pd.DataFrame) -> pd.DataFrame:
    profile_col = _profile_id_column(df)
    if not profile_col or profile_col not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["ds"] = pd.to_datetime(first_series(work, "ds"), errors="coerce")
    work[profile_col] = first_series(work, profile_col).astype(str).str.strip()
    work = work.dropna(subset=["ds"])
    work = work[(work[profile_col] != "") & (work[profile_col].str.lower() != "nan")]
    if work.empty:
        return pd.DataFrame()
    work["month"] = work["ds"].dt.to_period("M").dt.to_timestamp()
    rows = []
    total_rev = float(pd.to_numeric(work["y"], errors="coerce").sum() or 1.0)
    max_date = work["ds"].max()
    for profile, sub in work.groupby(profile_col):
        sub = sub.sort_values("ds").copy()
        monthly = sub.groupby("month", as_index=False).agg(monthly_revenue=("y", "sum"))
        lifetime = float(pd.to_numeric(sub["y"], errors="coerce").sum() or 0.0)
        recent_cut = max_date - pd.Timedelta(days=60)
        prior_cut = max_date - pd.Timedelta(days=120)
        recent = float(sub[sub["ds"] >= recent_cut]["y"].sum() or 0.0)
        prior = float(sub[(sub["ds"] < recent_cut) & (sub["ds"] >= prior_cut)]["y"].sum() or 0.0)
        freq = float(sub["ds"].nunique())
        recency = float((max_date - sub["ds"].max()).days)
        trend = (recent - prior) / max(abs(prior), 1.0)
        rows.append({
            "profile_id": str(profile),
            "lifetime_revenue": lifetime,
            "recent_revenue": recent,
            "prior_revenue": prior,
            "recent_spend_trend": trend,
            "recency_days": recency,
            "purchase_frequency": freq,
            "revenue_share": lifetime / total_rev,
            "months_active": int(monthly["month"].nunique()),
            "next_revenue_proxy": recent,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    whale_threshold = out["lifetime_revenue"].quantile(0.75)
    out = out[out["lifetime_revenue"] >= whale_threshold].copy()
    return out.sort_values("lifetime_revenue", ascending=False).reset_index(drop=True)


def _behavior_label(row: pd.Series) -> str:
    trend = float(row.get("recent_spend_trend") or 0.0)
    recency = float(row.get("recency_days") or 0.0)
    if trend <= -0.35 or recency >= 60:
        return "at risk"
    if trend <= -0.12:
        return "cooling"
    if trend >= 0.25 and recency <= 30:
        return "heating up"
    if trend > -0.35 and recency <= 45 and float(row.get("prior_revenue") or 0.0) > float(row.get("recent_revenue") or 0.0):
        return "recoverable"
    return "stable"


def _compute_whale_prediction_layer(df: pd.DataFrame, config: dict | None = None) -> dict:
    cfg = dict(config or {})
    frame = _profile_behavior_frame(df)
    if frame.empty or len(frame) < 3:
        return {"available": False, "error": "Profile-level history was not rich enough for whale behavior prediction."}
    features = ["lifetime_revenue", "recent_revenue", "prior_revenue", "recent_spend_trend", "recency_days", "purchase_frequency", "revenue_share", "months_active"]
    frame[features] = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    frame["cooling_target"] = ((frame["recent_spend_trend"] <= -0.15) | (frame["recency_days"] >= 45)).astype(int)
    frame["reactivation_target"] = ((frame["prior_revenue"] > frame["recent_revenue"]) & (frame["recency_days"] <= 90)).astype(int)
    frame["drop_target"] = (frame["recent_spend_trend"] <= -0.30).astype(int)
    models = {}
    metrics = {}
    for target in ["cooling_target", "reactivation_target", "drop_target"]:
        model, m = _fit_classifier(frame, features, target)
        models[target] = model
        metrics[target] = m
        _persist_artifact(cfg, f"lightgbm_whale_{target}", model, features, {
            "target_definition": target,
            "training_window": {"rows": int(len(frame))},
            "metrics": m,
        })
    scored = frame.copy()
    scored["cooling_risk"] = models["cooling_target"].predict_proba(scored[features])[:, 1] if models.get("cooling_target") is not None else scored["cooling_target"].astype(float)
    scored["reactivation_likelihood"] = models["reactivation_target"].predict_proba(scored[features])[:, 1] if models.get("reactivation_target") is not None else scored["reactivation_target"].astype(float)
    scored["revenue_drop_risk"] = models["drop_target"].predict_proba(scored[features])[:, 1] if models.get("drop_target") is not None else scored["drop_target"].astype(float)
    scored["stability_score"] = (1.0 - scored[["cooling_risk", "revenue_drop_risk"]].max(axis=1)).clip(0, 1)
    scored["behavior_label"] = scored.apply(_behavior_label, axis=1)
    scored.loc[(scored["cooling_risk"] >= 0.65) | (scored["revenue_drop_risk"] >= 0.65), "behavior_label"] = "at risk"
    scored.loc[(scored["reactivation_likelihood"] >= 0.65) & (scored["behavior_label"].isin(["at risk", "cooling"])), "behavior_label"] = "recoverable"
    scored["recommended_action"] = scored["behavior_label"].map({
        "stable": "Maintain current coverage and monitor concentration.",
        "at risk": "Prioritize retention outreach and investigate the recent value drop.",
        "cooling": "Review engagement and monetization before the profile becomes at risk.",
        "heating up": "Protect experience and consider expansion offers.",
        "recoverable": "Use win-back or targeted retention because prior value remains meaningful.",
    }).fillna("Monitor behavior.")
    show = scored.sort_values(["revenue_drop_risk", "cooling_risk", "lifetime_revenue"], ascending=False).copy()
    for c in ["cooling_risk", "reactivation_likelihood", "revenue_drop_risk", "stability_score", "revenue_share", "recent_spend_trend"]:
        show[c] = pd.to_numeric(show[c], errors="coerce").round(4)
    cols = ["profile_id", "behavior_label", "cooling_risk", "reactivation_likelihood", "revenue_drop_risk", "stability_score", "lifetime_revenue", "recent_revenue", "prior_revenue", "recency_days", "purchase_frequency", "revenue_share", "recommended_action"]
    return {"available": True, "scores": _records(show[cols], MAX_OUTPUT_ROWS), "artifact_profile": {"model": "LGBMClassifier/fallback", "training_rows": int(len(frame)), "metrics": metrics}}


def _latest_feature_frame(df: pd.DataFrame, config: dict) -> tuple[LGBMRegressor | None, pd.DataFrame, list[str], dict]:
    model, feat, features, error, fit_meta = _fit_explain_model(df, config)
    if model is None or feat.empty or not features:
        return None, pd.DataFrame(), [], {"error": error}
    latest = feat.sort_values("ds").tail(min(30, len(feat))).copy()
    return model, latest, features, fit_meta


def _safe_predict(model: LGBMRegressor, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    out = np.asarray(model.predict(frame[features]), dtype=float)
    return np.clip(out, 0.0, None)


def _compute_diminishing_returns_layer(df: pd.DataFrame, config: dict | None = None) -> dict:
    cfg = dict(config or {})
    if "cost" not in df.columns:
        return {"available": False, "error": "Spend/cost data is required for ML diminishing returns."}
    model, latest, features, fit_meta = _latest_feature_frame(df, cfg)
    if model is None or latest.empty or "cost" not in latest.columns:
        return {"available": False, "error": "A valid LightGBM revenue model with cost features was not available."}
    base_cost = float(pd.to_numeric(latest["cost"], errors="coerce").mean() or 0.0)
    if base_cost <= 0:
        return {"available": False, "error": "Cost values were not positive enough for spend response simulation."}
    rows = []
    multipliers = [0.70, 0.85, 1.0, 1.10, 1.25, 1.50]
    base_pred = float(_safe_predict(model, latest, features).mean())
    for mult in multipliers:
        sim = latest.copy()
        sim["cost"] = pd.to_numeric(sim["cost"], errors="coerce").fillna(base_cost) * mult
        if "cost_log1p" in sim.columns:
            sim["cost_log1p"] = np.log1p(sim["cost"].clip(lower=0))
        if "leads" in sim.columns:
            sim["leads"] = pd.to_numeric(sim["leads"], errors="coerce").fillna(0) * (1 + ((mult - 1) * 0.65))
            if "leads_log1p" in sim.columns:
                sim["leads_log1p"] = np.log1p(sim["leads"].clip(lower=0))
        pred = float(_safe_predict(model, sim, features).mean())
        spend = base_cost * mult
        rev_delta = pred - base_pred
        spend_delta = spend - base_cost
        marginal_roas = rev_delta / spend_delta if abs(spend_delta) > 1e-9 else np.nan
        roas = pred / spend if spend > 0 else np.nan
        if mult < 1.0:
            zone = "efficient zone" if pred >= base_pred * 0.9 else "under-invested"
        elif pd.notna(marginal_roas) and marginal_roas <= 0.25:
            zone = "overspend zone"
        elif pd.notna(marginal_roas) and marginal_roas < max(roas * 0.45, 0.5):
            zone = "saturation zone"
        else:
            zone = "efficient zone"
        label = "overspending" if zone == "overspend zone" else "approaching saturation" if zone == "saturation zone" else "still scalable"
        rows.append({
            "spend_change_pct": round((mult - 1) * 100.0, 1),
            "projected_revenue": round(pred, 2),
            "projected_roas": round(float(roas), 4) if pd.notna(roas) else None,
            "marginal_roas": round(float(marginal_roas), 4) if pd.notna(marginal_roas) else None,
            "zone": zone,
            "business_label": label,
        })
    headroom_rows = [r for r in rows if r["business_label"] == "still scalable" and r["spend_change_pct"] > 0]
    headroom = max([r["spend_change_pct"] for r in headroom_rows], default=0)
    return {"available": True, "response_curve": rows, "summary": {"scale_headroom_pct": headroom, "current_revenue_baseline": round(base_pred, 2), "risk_label": _risk_label(float(np.std([r["projected_revenue"] for r in rows]) / max(base_pred, 1.0)))}, "artifact_profile": fit_meta}


def _compute_scenario_simulation_layer(df: pd.DataFrame, config: dict | None = None) -> dict:
    cfg = dict(config or {})
    model, latest, features, fit_meta = _latest_feature_frame(df, cfg)
    if model is None or latest.empty:
        return {"available": False, "error": "A valid LightGBM revenue model was not available for simulation."}
    base_pred = float(_safe_predict(model, latest, features).mean())
    scenarios = [
        ("Current baseline", 1.00, 1.00, 1.00),
        ("Spend +10%", 1.10, 1.06, 1.00),
        ("Spend +20% with mix drag", 1.20, 1.08, 0.96),
        ("Lead volume +15%", 1.00, 1.15, 1.02),
        ("Affiliate allocation improvement", 1.05, 1.08, 1.05),
        ("Whale retention lift", 1.00, 1.02, 1.06),
    ]
    rows = []
    residual_std = float((pd.to_numeric(latest["y"], errors="coerce") - _safe_predict(model, latest, features)).std(ddof=0) or 0.0)
    for name, spend_mult, lead_mult, quality_mult in scenarios:
        sim = latest.copy()
        if "cost" in sim.columns:
            sim["cost"] = pd.to_numeric(sim["cost"], errors="coerce").fillna(0) * spend_mult
            if "cost_log1p" in sim.columns:
                sim["cost_log1p"] = np.log1p(sim["cost"].clip(lower=0))
        if "leads" in sim.columns:
            sim["leads"] = pd.to_numeric(sim["leads"], errors="coerce").fillna(0) * lead_mult
            if "leads_log1p" in sim.columns:
                sim["leads_log1p"] = np.log1p(sim["leads"].clip(lower=0))
        if "revenue_per_lead" in sim.columns:
            sim["revenue_per_lead"] = pd.to_numeric(sim["revenue_per_lead"], errors="coerce").fillna(0) * quality_mult
        pred = float(_safe_predict(model, sim, features).mean())
        cost = float(pd.to_numeric(sim["cost"], errors="coerce").mean() or 0.0) if "cost" in sim.columns else 0.0
        low = max(0.0, pred - residual_std)
        high = pred + residual_std
        warning = ""
        if pred > base_pred * 4 or pred < 0:
            warning = "Scenario produced an extreme result; treat as directional only."
        rows.append({
            "scenario": name,
            "projected_revenue": round(pred, 2),
            "projected_roas": round(pred / cost, 4) if cost > 0 else None,
            "projected_lead_quality_effect": round((quality_mult - 1.0) * 100.0, 1),
            "risk_band_low": round(low, 2),
            "risk_band_high": round(high, 2),
            "interpretation": "Upside scenario" if pred > base_pred * 1.03 else "Downside/neutral scenario" if pred < base_pred * 0.97 else "Near baseline",
            "warning": warning,
        })
    return {"available": True, "scenarios": rows, "artifact_profile": fit_meta}


def _compute_affiliate_quality_layer(df: pd.DataFrame, config: dict | None = None) -> dict:
    frame = _dimension_month_frame(df)
    if frame.empty or "affiliate" not in set(frame["dimension_type"].astype(str)):
        return {"available": False, "error": "Affiliate dimensions were not available for ML affiliate quality scoring."}
    aff = frame[frame["dimension_type"] == "affiliate"].copy()
    if aff.empty:
        return {"available": False, "error": "No affiliate rows available."}
    rows = []
    for affiliate, sub in aff.groupby("dimension"):
        sub = sub.sort_values("month")
        revenue = pd.to_numeric(sub["y"], errors="coerce").fillna(0)
        recent = float(revenue.tail(3).mean() or 0.0)
        prior = float(revenue.head(max(len(revenue) - 3, 1)).mean() or 0.0)
        volatility = float(revenue.std(ddof=0) or 0.0)
        consistency = recent / max(volatility, 1.0)
        roas = float(pd.to_numeric(sub.get("roas_calc", pd.Series(dtype=float)), errors="coerce").tail(3).mean() or 0.0)
        rpl = float(pd.to_numeric(sub.get("rpl_calc", pd.Series(dtype=float)), errors="coerce").tail(3).mean() or 0.0)
        trend = (recent - prior) / max(abs(prior), 1.0)
        short_strong_long_weak = bool(recent > prior * 1.15 and trend < 0.1 and volatility > max(recent * 0.75, 1.0))
        score = (np.tanh(recent / max(float(aff["y"].mean() or 1.0), 1.0)) * 35) + (np.tanh(roas) * 25) + (np.tanh(consistency / 4) * 25) + (np.tanh(rpl / max(float(pd.to_numeric(aff.get("rpl_calc", pd.Series([1])), errors="coerce").median() or 1.0), 1.0)) * 15)
        score = float(np.clip(score, 0, 100))
        tier = "premium" if score >= 75 else "growth" if score >= 55 else "watch" if score >= 35 else "weak"
        rec = "scale" if tier == "premium" and not short_strong_long_weak else "test" if tier == "growth" else "caution" if short_strong_long_weak or tier == "weak" else "monitor"
        rows.append({
            "affiliate": str(affiliate),
            "quality_score": round(score, 2),
            "quality_tier": tier,
            "recommendation": rec,
            "recent_revenue": round(recent, 2),
            "trend_pct": round(trend * 100.0, 2),
            "consistency_score": round(consistency, 3),
            "recent_roas": round(roas, 4),
            "recent_rpl": round(rpl, 4),
            "short_term_strong_long_term_weak": short_strong_long_weak,
        })
    return {"available": True, "scores": sorted(rows, key=lambda r: r["quality_score"], reverse=True)[:MAX_OUTPUT_ROWS]}


def _compute_confidence_layer(df: pd.DataFrame, ranking_layer: dict, simulation_layer: dict, returns_layer: dict, shap_layer: dict) -> dict:
    y = pd.to_numeric(df.get("y", pd.Series(dtype=float)), errors="coerce").dropna()
    volatility = float(y.tail(min(30, len(y))).std(ddof=0) / max(abs(float(y.tail(min(30, len(y))).mean() or 1.0)), 1.0)) if not y.empty else 1.0
    risk_score = float(np.clip(volatility, 0, 1))
    rows = [{
        "surface": "forecast / revenue model",
        "risk_label": _risk_label(risk_score),
        "confidence_low": round(max(0.0, 1 - risk_score - 0.15), 3),
        "confidence_high": round(max(0.0, 1 - risk_score + 0.15), 3),
        "reason": "Recent revenue volatility drives this confidence label.",
    }]
    if ranking_layer.get("available"):
        rows.append({"surface": "rankings", "risk_label": "moderate risk", "confidence_low": 0.55, "confidence_high": 0.8, "reason": "Rankings are strongest where dimensions have repeated history."})
    if simulation_layer.get("available"):
        rows.append({"surface": "simulations", "risk_label": "moderate risk", "confidence_low": 0.5, "confidence_high": 0.78, "reason": "Scenario outputs are directional and bounded by observed model behavior."})
    if returns_layer.get("available"):
        rows.append({"surface": "diminishing returns", "risk_label": returns_layer.get("summary", {}).get("risk_label", "moderate risk"), "confidence_low": 0.48, "confidence_high": 0.76, "reason": "Response curves depend on observed spend variation."})
    return {"available": True, "risk_rows": rows, "summary": rows[0]}


def build_lightgbm_intelligence(df: pd.DataFrame, config: dict | None = None) -> dict:
    if df is None or df.empty:
        return {"available": False, "error": "No modeling data was available for LightGBM intelligence."}
    cfg = dict(config or {})
    work = dedupe_columns_keep_first(df.copy())
    if not {"ds", "y"}.issubset(work.columns):
        return {"available": False, "error": "LightGBM intelligence requires normalized ds/y columns."}
    work["ds"] = pd.to_datetime(first_series(work, "ds"), errors="coerce")
    work["y"] = pd.to_numeric(first_series(work, "y"), errors="coerce")
    work = work.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
    if "series_id" not in work.columns:
        work["series_id"] = "company_total"
    if "category" not in work.columns:
        work["category"] = work["series_id"].astype(str)
    if len(work) < 8:
        return {"available": False, "error": "At least 8 usable periods are needed for LightGBM intelligence."}

    shap_layer = _compute_shap_layer(work, cfg) if _enabled(cfg, "enable_shap_root_cause") else {"available": False, "disabled": True}
    ranking_layer = _compute_ranking_layer(work, cfg) if _enabled(cfg, "enable_ranking_engine") else {"available": False, "disabled": True}
    lead_quality_layer = _compute_lead_quality_layer(work, cfg) if _enabled(cfg, "enable_lead_quality_scoring") else {"available": False, "disabled": True}
    whale_layer = _compute_whale_prediction_layer(work, cfg) if _enabled(cfg, "enable_whale_prediction") else {"available": False, "disabled": True}
    anomaly_layer = _compute_residual_anomaly_layer(work, cfg, shap_layer=shap_layer) if _enabled(cfg, "enable_model_anomaly_detection") else {"available": False, "disabled": True}
    diminishing_returns_layer = _compute_diminishing_returns_layer(work, cfg) if _enabled(cfg, "enable_ml_diminishing_returns") else {"available": False, "disabled": True}
    simulation_layer = _compute_scenario_simulation_layer(work, cfg) if _enabled(cfg, "enable_scenario_simulation") else {"available": False, "disabled": True}
    affiliate_quality_layer = _compute_affiliate_quality_layer(work, cfg) if _enabled(cfg, "enable_affiliate_quality_score") else {"available": False, "disabled": True}
    confidence_layer = _compute_confidence_layer(work, ranking_layer, simulation_layer, diminishing_returns_layer, shap_layer) if _enabled(cfg, "enable_confidence_layer") else {"available": False, "disabled": True}

    summaries = []
    if shap_layer.get("available"):
        summaries.extend(shap_layer.get("executive_summary") or [])
    if ranking_layer.get("available") and ranking_layer.get("ranked_opportunities"):
        top = ranking_layer["ranked_opportunities"][0]
        summaries.append(f"Top LightGBM-ranked opportunity: {top.get('dimension')} is marked {top.get('recommendation')}.")
    if lead_quality_layer.get("available") and lead_quality_layer.get("scores"):
        top_lq = lead_quality_layer["scores"][0]
        summaries.append(f"Lead quality scoring highlights {top_lq.get('dimension')} as {top_lq.get('confidence_bucket')} confidence and {top_lq.get('value_tier')} value.")
    if whale_layer.get("available") and whale_layer.get("scores"):
        risky = [r for r in whale_layer["scores"] if r.get("behavior_label") in {"at risk", "cooling", "recoverable"}]
        summaries.append(f"Whale prediction found {len(risky)} high-value user profiles needing attention.")
    if diminishing_returns_layer.get("available"):
        label = (diminishing_returns_layer.get("response_curve") or [{}])[-1].get("business_label")
        summaries.append(f"ML diminishing returns currently reads as {label or 'directional'} at the high-spend scenario.")
    if affiliate_quality_layer.get("available") and affiliate_quality_layer.get("scores"):
        top_aff = affiliate_quality_layer["scores"][0]
        summaries.append(f"Affiliate quality scoring ranks {top_aff.get('affiliate')} as {top_aff.get('quality_tier')} with a {top_aff.get('recommendation')} recommendation.")

    return {
        "available": any(layer.get("available") for layer in [shap_layer, ranking_layer, lead_quality_layer, whale_layer, anomaly_layer, diminishing_returns_layer, simulation_layer, affiliate_quality_layer, confidence_layer]),
        "executive_summary": summaries[:8],
        "shap": shap_layer,
        "ranking": ranking_layer,
        "lead_quality": lead_quality_layer,
        "whale_prediction": whale_layer,
        "residual_anomalies": anomaly_layer,
        "diminishing_returns": diminishing_returns_layer,
        "scenario_simulation": simulation_layer,
        "affiliate_quality": affiliate_quality_layer,
        "confidence": confidence_layer,
        "config": {key: _enabled(cfg, key) for key in DEFAULT_TOGGLES},
    }
