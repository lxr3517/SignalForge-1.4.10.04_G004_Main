
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


DATE_CANDIDATES = ["tran_date", "transaction_date", "purchase_date", "date", "ds"]
MEMBER_CANDIDATES = ["member_id", "profile_id", "user_id", "customer_id", "account_id"]
REVENUE_CANDIDATES = ["revenue", "amount", "net_revenue", "sales", "y"]
PURCHASE_ID_CANDIDATES = ["purchase_id", "transaction_id", "order_id", "tran_id"]
LEAD_CANDIDATES = ["new_leads", "leads", "lead_count", "lead_volume", "lead_qty"]


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lowered:
            return lowered[c.lower()]
    return None


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    try:
        n = float(numerator)
        d = float(denominator)
        if not np.isfinite(n) or not np.isfinite(d) or d == 0:
            return None
        return n / d
    except Exception:
        return None


def _json_records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    temp = df.copy()
    if limit is not None:
        temp = temp.head(limit)
    for col in temp.columns:
        if pd.api.types.is_datetime64_any_dtype(temp[col]):
            temp[col] = temp[col].dt.strftime('%Y-%m-%d')
    return temp.replace([np.inf, -np.inf], np.nan).where(pd.notna(temp), None).to_dict(orient="records")


def build_spending_slowdown(df: pd.DataFrame | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": False,
        "message": "",
        "cards": {},
        "charts": [],
        "decomposition_table": [],
        "returning_table": [],
        "cohort_table": [],
        "transaction_columns": {},
    }

    if df is None or df.empty:
        out["message"] = "No source rows were available for spending diagnostics."
        return out

    raw = df.copy()

    date_col = _pick_column(raw, DATE_CANDIDATES)
    member_col = _pick_column(raw, MEMBER_CANDIDATES)
    revenue_col = _pick_column(raw, REVENUE_CANDIDATES)
    purchase_id_col = _pick_column(raw, PURCHASE_ID_CANDIDATES)
    lead_col = _pick_column(raw, LEAD_CANDIDATES)

    out["transaction_columns"] = {
        "date_col": date_col,
        "member_col": member_col,
        "revenue_col": revenue_col,
        "purchase_id_col": purchase_id_col,
        "lead_col": lead_col,
    }

    if not date_col or not revenue_col:
        out["message"] = (
            "Spending Slowdown is hidden because the mapped data does not contain the date and revenue columns needed "
            "for this view."
        )
        return out
    if not member_col and not lead_col:
        out["message"] = (
            "Spending Slowdown is hidden because neither purchaser/profile ids nor a leads column were found. "
            "Map a profile id to count unique purchasers, or map the leads sheet so the view can fall back to new leads."
        )
        return out

    using_profiles = bool(member_col)
    entity_label = "Unique purchasers" if using_profiles else "New leads"
    entity_label_singular = "purchaser" if using_profiles else "new lead"
    entity_trend_title = "Unique Purchasers" if using_profiles else "New Leads"
    entity_y_title = "Purchasers" if using_profiles else "New leads"
    activity_label = "Rolling Purchases / Purchaser" if using_profiles else "Purchases / New Lead"
    revenue_per_entity_label = "Revenue / Purchaser" if using_profiles else "Revenue / New Lead"
    entity_component_label = "Purchaser count" if using_profiles else "New leads"
    fallback_message = (
        "Using unique purchaser/profile ids from the revenue data."
        if using_profiles
        else "Profile ids were not mapped, so this view uses new leads from the leads sheet instead of purchaser counts."
    )

    work = raw[[c for c in [date_col, member_col, revenue_col, purchase_id_col, lead_col] if c and c in raw.columns]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[revenue_col] = pd.to_numeric(work[revenue_col], errors="coerce")
    if lead_col and lead_col in work.columns:
        work[lead_col] = pd.to_numeric(work[lead_col], errors="coerce")
    drop_subset = [date_col, revenue_col]
    if using_profiles:
        drop_subset.append(member_col)
    work = work.dropna(subset=drop_subset).copy()
    if work.empty:
        out["message"] = "Rows were found, but none had usable date/revenue and purchaser or leads values after cleaning."
        return out

    work["date"] = work[date_col].dt.floor("D")
    if using_profiles:
        work["member_key"] = work[member_col].astype(str)
        if purchase_id_col and purchase_id_col in work.columns:
            purchase_count = work.groupby(["date", "member_key"])[purchase_id_col].nunique(dropna=True).reset_index(name="purchases")
        else:
            purchase_count = work.groupby(["date", "member_key"]).size().reset_index(name="purchases")

        user_day = (
            work.groupby(["date", "member_key"], as_index=False)
            .agg(revenue=(revenue_col, "sum"))
            .merge(purchase_count, on=["date", "member_key"], how="left")
        )
        user_day["purchases"] = pd.to_numeric(user_day["purchases"], errors="coerce").fillna(0)
        user_day["is_active"] = (user_day["revenue"] > 0).astype(int)

        first_purchase = (
            user_day.groupby("member_key", as_index=False)["date"]
            .min()
            .rename(columns={"date": "first_purchase_date"})
        )
        user_day = user_day.merge(first_purchase, on="member_key", how="left")
        user_day["user_age_days"] = (user_day["date"] - user_day["first_purchase_date"]).dt.days
        user_day["is_returning"] = user_day["user_age_days"] > 0
    else:
        user_day = pd.DataFrame(columns=["date", "member_key", "revenue", "purchases", "user_age_days", "is_returning"])

    def _cohort_bucket(age: float) -> str:
        if pd.isna(age):
            return "Unknown"
        age = int(age)
        if age <= 30:
            return "0-30"
        if age <= 90:
            return "31-90"
        if age <= 180:
            return "91-180"
        return "180+"

    if using_profiles:
        user_day["cohort"] = user_day["user_age_days"].apply(_cohort_bucket)
        daily = (
            user_day.groupby("date", as_index=False)
            .agg(
                entity_count=("member_key", "nunique"),
                revenue=("revenue", "sum"),
                purchases=("purchases", "sum"),
            )
            .sort_values("date")
        )
    else:
        lead_daily = (
            work.groupby("date", as_index=False)
            .agg(
                entity_count=(lead_col, "sum"),
                revenue=(revenue_col, "sum"),
                purchases=(purchase_id_col, "nunique") if purchase_id_col and purchase_id_col in work.columns else (revenue_col, "size"),
            )
            .sort_values("date")
        )
        daily = lead_daily.copy()
    daily["rev_per_entity"] = daily.apply(lambda r: _safe_ratio(r["revenue"], r["entity_count"]), axis=1)
    daily["purchases_per_entity"] = daily.apply(lambda r: _safe_ratio(r["purchases"], r["entity_count"]), axis=1)
    daily["avg_order_value"] = daily.apply(lambda r: _safe_ratio(r["revenue"], r["purchases"]), axis=1)

    returning_daily = (
        user_day[user_day["is_returning"]]
        .groupby("date", as_index=False)
        .agg(
            returning_spenders=("member_key", "nunique"),
            returning_revenue=("revenue", "sum"),
            returning_purchases=("purchases", "sum"),
        )
        .sort_values("date")
        if using_profiles
        else pd.DataFrame()
    )
    if not returning_daily.empty:
        returning_daily["revenue_per_returning_user"] = returning_daily.apply(
            lambda r: _safe_ratio(r["returning_revenue"], r["returning_spenders"]), axis=1
        )
        returning_daily["purchases_per_returning_user"] = returning_daily.apply(
            lambda r: _safe_ratio(r["returning_purchases"], r["returning_spenders"]), axis=1
        )

    cards = {
        "identity_mode": "profiles" if using_profiles else "leads",
        "identity_message": fallback_message,
        "entity_label": entity_label,
        "entity_label_singular": entity_label_singular,
        "entity_count_avg": float(pd.to_numeric(daily["entity_count"], errors="coerce").mean() or 0.0),
        "revenue_per_entity_avg": float(pd.to_numeric(daily["rev_per_entity"], errors="coerce").mean() or 0.0),
        "purchases_per_entity_avg": float(pd.to_numeric(daily["purchases_per_entity"], errors="coerce").mean() or 0.0),
        "avg_order_value_avg": float(pd.to_numeric(daily["avg_order_value"], errors="coerce").mean() or 0.0),
        "total_revenue": float(pd.to_numeric(daily["revenue"], errors="coerce").sum() or 0.0),
        "returning_revenue_share": float(
            (
                pd.to_numeric(returning_daily["returning_revenue"], errors="coerce").sum()
                / pd.to_numeric(daily["revenue"], errors="coerce").sum()
            )
            if not returning_daily.empty and float(pd.to_numeric(daily["revenue"], errors="coerce").sum() or 0.0) > 0
            else 0.0
        ),
    }

    # decomposition: recent 7 days vs prior 30 days
    recent = daily.tail(min(len(daily), 7)).copy()
    baseline_pool = daily.iloc[:-len(recent)] if len(daily) > len(recent) else pd.DataFrame()
    baseline = baseline_pool.tail(30) if not baseline_pool.empty else daily.tail(min(len(daily), 30))
    if not recent.empty and not baseline.empty:
        baseline_active = float(baseline["entity_count"].mean() or 0.0)
        baseline_freq = float(pd.to_numeric(baseline["purchases_per_entity"], errors="coerce").mean() or 0.0)
        baseline_aov = float(pd.to_numeric(baseline["avg_order_value"], errors="coerce").mean() or 0.0)

        current_active = float(recent["entity_count"].mean() or 0.0)
        current_freq = float(pd.to_numeric(recent["purchases_per_entity"], errors="coerce").mean() or 0.0)
        current_aov = float(pd.to_numeric(recent["avg_order_value"], errors="coerce").mean() or 0.0)

        impact_spenders = (current_active - baseline_active) * baseline_freq * baseline_aov
        impact_frequency = current_active * (current_freq - baseline_freq) * baseline_aov
        impact_aov = current_active * current_freq * (current_aov - baseline_aov)

        decomposition = pd.DataFrame(
            [
                {"component": entity_component_label, "impact_revenue": impact_spenders},
                {"component": activity_label, "impact_revenue": impact_frequency},
                {"component": "Order value", "impact_revenue": impact_aov},
            ]
        )
        out["decomposition_table"] = _json_records(decomposition.round(2))
    else:
        decomposition = pd.DataFrame(columns=["component", "impact_revenue"])

    cohort_table = pd.DataFrame()
    if using_profiles:
        cohort_table = (
            user_day.groupby("cohort", as_index=False)
            .agg(
                users=("member_key", "nunique"),
                revenue=("revenue", "sum"),
                purchases=("purchases", "sum"),
            )
            .copy()
        )
        if not cohort_table.empty:
            cohort_table["revenue_per_user"] = cohort_table.apply(lambda r: _safe_ratio(r["revenue"], r["users"]), axis=1)
            cohort_table["purchases_per_user"] = cohort_table.apply(lambda r: _safe_ratio(r["purchases"], r["users"]), axis=1)
            order = pd.CategoricalDtype(["0-30", "31-90", "91-180", "180+", "Unknown"], ordered=True)
            cohort_table["cohort"] = cohort_table["cohort"].astype(order)
            cohort_table = cohort_table.sort_values("cohort")
            cohort_table["cohort"] = cohort_table["cohort"].astype(str)
    out["cohort_table"] = _json_records(cohort_table.round(4))

    if not returning_daily.empty:
        returning_view = returning_daily.copy()
        returning_view["date"] = pd.to_datetime(returning_view["date"])
        out["returning_table"] = _json_records(returning_view.tail(60).round(4))
        cards["returning_spenders_avg"] = float(pd.to_numeric(returning_daily["returning_spenders"], errors="coerce").mean() or 0.0)
        cards["returning_revenue_avg"] = float(pd.to_numeric(returning_daily["returning_revenue"], errors="coerce").mean() or 0.0)
        cards["revenue_per_returning_user_avg"] = float(pd.to_numeric(returning_daily["revenue_per_returning_user"], errors="coerce").mean() or 0.0)
        cards["purchases_per_returning_user_avg"] = float(pd.to_numeric(returning_daily["purchases_per_returning_user"], errors="coerce").mean() or 0.0)

    charts = [
        {
            "key": "revenue",
            "title": "Revenue Trend",
            "subtitle": "Daily revenue from transaction-level rows.",
            "x": daily["date"].dt.strftime("%Y-%m-%d").tolist(),
            "y": pd.to_numeric(daily["revenue"], errors="coerce").round(2).tolist(),
            "y_title": "Revenue",
            "hover": "Revenue: $%{y:,.0f}<extra></extra>",
        },
        {
            "key": "entity_count",
            "title": entity_trend_title,
            "subtitle": (
                "How many unique purchasers were active each day."
                if using_profiles
                else "How many new leads were available each day from the mapped leads data."
            ),
            "x": daily["date"].dt.strftime("%Y-%m-%d").tolist(),
            "y": pd.to_numeric(daily["entity_count"], errors="coerce").round(2).tolist(),
            "y_title": entity_y_title,
            "hover": (
                "Unique purchasers: %{y:,.0f}<extra></extra>"
                if using_profiles
                else "New leads: %{y:,.0f}<extra></extra>"
            ),
        },
        {
            "key": "rev_per_entity",
            "title": revenue_per_entity_label,
            "subtitle": (
                "Average revenue per unique purchaser by day."
                if using_profiles
                else "Average revenue supported by each new lead by day."
            ),
            "x": daily["date"].dt.strftime("%Y-%m-%d").tolist(),
            "y": pd.to_numeric(daily["rev_per_entity"], errors="coerce").round(2).tolist(),
            "y_title": revenue_per_entity_label,
            "hover": f"{revenue_per_entity_label}: $%{{y:,.2f}}<extra></extra>",
        },
        {
            "key": "purchases_per_entity",
            "title": activity_label,
            "subtitle": (
                "Average purchase count per unique purchaser."
                if using_profiles
                else "Average purchase count supported by each new lead."
            ),
            "x": daily["date"].dt.strftime("%Y-%m-%d").tolist(),
            "y": pd.to_numeric(daily["purchases_per_entity"], errors="coerce").round(3).tolist(),
            "y_title": activity_label,
            "hover": f"{activity_label}: %{{y:.2f}}<extra></extra>",
        },
        {
            "key": "avg_order_value",
            "title": "Average Order Value",
            "subtitle": "Average revenue per purchase by day.",
            "x": daily["date"].dt.strftime("%Y-%m-%d").tolist(),
            "y": pd.to_numeric(daily["avg_order_value"], errors="coerce").round(2).tolist(),
            "y_title": "Avg Order Value",
            "hover": "Average order value: $%{y:,.2f}<extra></extra>",
        },
    ]

    if not decomposition.empty:
        charts.append(
            {
                "key": "decomposition",
                "title": "Revenue Change Decomposition",
                "subtitle": "Recent 7-day average versus the prior 30-day baseline.",
                "type": "bar",
                "x": decomposition["component"].tolist(),
                "y": decomposition["impact_revenue"].round(2).tolist(),
                "y_title": "Impact on Revenue",
                "hover": "Impact: $%{y:,.0f}<extra></extra>",
            }
        )

    out["available"] = True
    out["cards"] = cards
    out["charts"] = charts
    out["message"] = fallback_message
    return out
