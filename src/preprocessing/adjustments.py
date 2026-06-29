from __future__ import annotations

import pandas as pd


def apply_manual_adjustments(
    df: pd.DataFrame,
    exclude_start=None,
    exclude_end=None,
    mark_outage: bool = False,
    mark_promo: bool = False,
    use_holidays: bool = False,
    lagged_logic: bool = True,
    partial_mode: str = "flag_only",
) -> pd.DataFrame:
    adjusted = df.copy()
    adjusted["outage_flag"] = adjusted.get("outage_flag", 0)
    adjusted["promo_flag"] = adjusted.get("promo_flag", 0)
    adjusted["holiday_flag"] = adjusted.get("holiday_flag", 0)

    if exclude_start and exclude_end:
        start = pd.to_datetime(exclude_start)
        end = pd.to_datetime(exclude_end)
        mask = (adjusted["ds"] >= start) & (adjusted["ds"] <= end)
        if mark_outage:
            adjusted.loc[mask, "outage_flag"] = 1
        if mark_promo:
            adjusted.loc[mask, "promo_flag"] = 1
        if partial_mode == "exclude":
            adjusted = adjusted.loc[~mask].copy()

    if use_holidays:
        adjusted["holiday_flag"] = adjusted["holiday_flag"].fillna(0)

    adjusted["lagged_logic_enabled"] = int(lagged_logic)
    adjusted["partial_mode"] = partial_mode
    return adjusted.reset_index(drop=True)
