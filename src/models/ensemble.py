from __future__ import annotations

import pandas as pd


def make_scenario_frame(blended_forecast: list[dict]) -> list[dict]:
    if not blended_forecast:
        return []
    df = pd.DataFrame(blended_forecast).copy()
    if not {"ds", "yhat"}.issubset(df.columns):
        return []
    expected = pd.to_numeric(df["yhat"], errors="coerce").clip(lower=0)
    if {"yhat_lower", "yhat_upper"}.issubset(df.columns):
        conservative = pd.to_numeric(df["yhat_lower"], errors="coerce").clip(lower=0)
        aggressive   = pd.to_numeric(df["yhat_upper"], errors="coerce").clip(lower=0)
    else:
        conservative = expected * 0.92
        aggressive   = expected * 1.08
    # Guarantee conservative <= expected <= aggressive
    conservative = conservative.clip(upper=expected)
    aggressive   = aggressive.clip(lower=expected)
    conservative = conservative.where(conservative.notna(), expected * 0.92)
    aggressive   = aggressive.where(aggressive.notna(),   expected * 1.08)
    return pd.DataFrame({
        "ds": df["ds"],
        "conservative": conservative.round(2),
        "expected":     expected.round(2),
        "aggressive":   aggressive.round(2),
    }).to_dict(orient="records")
