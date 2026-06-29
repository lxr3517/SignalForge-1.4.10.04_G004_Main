from __future__ import annotations

import pandas as pd


def build_narrative(df: pd.DataFrame, decision: dict, config: dict) -> str:
    best = decision.get("best_model", {}).get("model", "unknown")
    target = config.get("target", "revenue")
    horizon = config.get("horizon", "30D")
    latest_mean = df["y"].tail(min(30, len(df))).mean()
    prior_mean = df["y"].tail(min(60, len(df))).head(min(30, len(df))).mean()

    trend_phrase = "stable"
    if latest_mean > prior_mean:
        trend_phrase = "improving"
    elif latest_mean < prior_mean:
        trend_phrase = "softening"

    return (
        f"The selected forecast emphasizes {target} over the next {horizon}. "
        f"The best single model was {best}, while the blended view combines the top performers. "
        f"Recent observed performance appears {trend_phrase}, and the app uses that recent pattern as part of its ranking logic."
    )
