from __future__ import annotations


def extract_feature_importance(metrics: list[dict]) -> dict:
    for metric in metrics:
        if metric.get("model") == "lightgbm" and "feature_importance" in metric:
            return metric["feature_importance"]
    return {}
