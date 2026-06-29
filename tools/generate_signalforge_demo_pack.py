from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd


START_DATE = "2025-01-01"
END_DATE = "2026-03-31"
SEED = 20260424


@dataclass(frozen=True)
class AdSpec:
    platform: str
    campaign: str
    ad_name: str
    affiliate: str
    base_daily_leads: float
    cpl: float
    conversion_rate: float
    whale_rate: float
    revenue_scale: float


AD_SPECS: tuple[AdSpec, ...] = (
    AdSpec("Google Search", "Brand Search", "Brand Core RSA", "AFF-1201", 18, 84, 0.42, 0.030, 1.10),
    AdSpec("Google Search", "Nonbrand Search", "High Intent Terms", "AFF-1202", 22, 93, 0.34, 0.025, 1.00),
    AdSpec("Google Display", "Prospecting Display", "In-Market Display A", "AFF-2201", 10, 58, 0.18, 0.014, 0.72),
    AdSpec("Google Display", "Retargeting Display", "Retargeting Banner 6", "AFF-2202", 8, 47, 0.28, 0.020, 0.88),
    AdSpec("Bing", "Brand Search", "Bing Brand Exact", "AFF-3201", 11, 76, 0.40, 0.026, 0.98),
    AdSpec("Bing", "Competitor Search", "Conquest Phrase Match", "AFF-3202", 9, 81, 0.29, 0.022, 0.90),
    AdSpec("Facebook", "Retargeting Social", "Remarketing Carousel", "AFF-4201", 17, 61, 0.26, 0.020, 0.82),
    AdSpec("Facebook", "Prospecting Social", "Lookalike Video v2", "AFF-4202", 19, 67, 0.20, 0.018, 0.78),
    AdSpec("LinkedIn", "B2B Decision Makers", "Director Persona Static", "AFF-5201", 7, 118, 0.31, 0.024, 1.18),
    AdSpec("LinkedIn", "ABM Outreach", "Enterprise ABM Lead Gen", "AFF-5202", 6, 132, 0.36, 0.028, 1.25),
    AdSpec("YouTube", "Explainer Video", "Demo Reel 30s", "AFF-6201", 5, 72, 0.15, 0.015, 0.69),
    AdSpec("YouTube", "Testimonial Video", "Customer Story Cutdown", "AFF-6202", 4, 70, 0.17, 0.016, 0.74),
    AdSpec("X / Twitter", "Event Push", "Launch Countdown Clip", "AFF-7201", 4, 56, 0.13, 0.011, 0.66),
    AdSpec("X / Twitter", "Thought Leadership", "Analyst Thread Promo", "AFF-7202", 3, 52, 0.14, 0.012, 0.71),
)


def _seasonality(day_index: int, date: pd.Timestamp) -> float:
    annual = 1.0 + 0.24 * math.sin((2 * math.pi * day_index / 365.25) - 0.75)
    weekly = 0.94 if date.weekday() >= 5 else 1.03
    quarter = 1.11 if date.month in (3, 6, 9, 11) else 0.98
    trend = 0.90 + 0.22 * (day_index / max((pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days, 1))
    return annual * weekly * quarter * trend


def _lead_count(rng: np.random.Generator, spec: AdSpec, day_index: int, date: pd.Timestamp) -> int:
    lam = spec.base_daily_leads * _seasonality(day_index, date)
    if date.month == 11:
        lam *= 1.16
    if date.month == 1:
        lam *= 0.92
    return max(0, int(rng.poisson(max(lam, 0.2))))


def _transaction_count(rng: np.random.Generator, converted: bool, whale: bool) -> int:
    if not converted:
        return 0
    if whale:
        return int(rng.integers(2, 6))
    return int(rng.choice([1, 1, 1, 2, 2, 3]))


def _revenue_value(rng: np.random.Generator, spec: AdSpec, whale: bool) -> float:
    base = rng.lognormal(mean=4.7 if not whale else 6.0, sigma=0.42 if not whale else 0.52)
    scaled = base * spec.revenue_scale
    if whale:
        scaled *= rng.uniform(2.2, 4.8)
    return round(max(scaled, 12.0), 2)


def build_demo_pack() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    end_ts = pd.Timestamp(END_DATE)

    lead_rows: list[dict] = []
    cost_rows: list[dict] = []
    revenue_rows: list[dict] = []
    cohort_rows: list[dict] = []

    profile_id = 990000001

    for day_index, date in enumerate(dates):
        for spec in AD_SPECS:
            leads = _lead_count(rng, spec, day_index, date)
            if leads <= 0:
                continue

            noise = rng.uniform(0.92, 1.11)
            day_cost = round(leads * spec.cpl * noise, 2)

            lead_rows.append(
                {
                    "Lead Date": date.strftime("%Y-%m-%d"),
                    "Platform": spec.platform,
                    "Campaign": spec.campaign,
                    "Ad Name": spec.ad_name,
                    "Aff_param": spec.affiliate,
                    "Count(distinct profile_id)": leads,
                }
            )
            cost_rows.append(
                {
                    "lead_day": date.strftime("%Y-%m-%d"),
                    "Platform": spec.platform,
                    "Campaign": spec.campaign,
                    "Ad Name": spec.ad_name,
                    "Aff_param": spec.affiliate,
                    "Cost": day_cost,
                }
            )

            for _ in range(leads):
                current_profile = profile_id
                profile_id += 1

                conversion_boost = 1.04 if date.month in (2, 3, 10, 11) else 0.97
                converted = bool(rng.random() < min(spec.conversion_rate * conversion_boost, 0.88))
                whale = converted and bool(rng.random() < spec.whale_rate)
                tx_count = _transaction_count(rng, converted, whale)

                if tx_count == 0:
                    continue

                lead_month = date.strftime("%Y-%b")
                lag_days = np.sort(rng.integers(0, 210 if whale else 150, size=tx_count))

                for lag in lag_days:
                    revenue_date = date + pd.Timedelta(days=int(lag))
                    if revenue_date > end_ts:
                        continue

                    revenue_value = _revenue_value(rng, spec, whale)
                    revenue_rows.append(
                        {
                            "Revenue Date": revenue_date.strftime("%Y-%m-%d"),
                            "Revenue": revenue_value,
                            "Platform": spec.platform,
                            "Campaign": spec.campaign,
                            "Ad Name": spec.ad_name,
                            "Aff_param": spec.affiliate,
                            "profile_id": current_profile,
                        }
                    )
                    cohort_rows.append(
                        {
                            "Lead Month": lead_month,
                            "Revenue Month": revenue_date.strftime("%Y-%b"),
                            "Revenue": revenue_value,
                            "Platform": spec.platform,
                            "Campaign": spec.campaign,
                            "Ad Name": spec.ad_name,
                            "Aff_param": spec.affiliate,
                            "profile_id": current_profile,
                        }
                    )

    return {
        "revenue": pd.DataFrame(revenue_rows).sort_values(
            ["Revenue Date", "Platform", "Campaign", "Ad Name", "profile_id"]
        ),
        "leads": pd.DataFrame(lead_rows).sort_values(
            ["Lead Date", "Platform", "Campaign", "Ad Name"]
        ),
        "cost": pd.DataFrame(cost_rows).sort_values(
            ["lead_day", "Platform", "Campaign", "Ad Name"]
        ),
        "cohort": pd.DataFrame(cohort_rows).sort_values(
            ["Lead Month", "Revenue Month", "Platform", "Campaign", "Ad Name", "profile_id"]
        ),
    }


def write_outputs(frames: dict[str, pd.DataFrame]) -> None:
    root = Path(__file__).resolve().parents[1]
    csv_dir = root / "tools" / "Data" / "CSV"
    xlsx_dir = root / "tools" / "Data" / "XL"
    csv_dir.mkdir(parents=True, exist_ok=True)
    xlsx_dir.mkdir(parents=True, exist_ok=True)

    name_map = {
        "revenue": "Demo_Revenue_Jan2025_to_Mar2026",
        "leads": "Demo_Leads_Jan2025_to_Mar2026",
        "cost": "Demo_Cost_Jan2025_to_Mar2026",
        "cohort": "Demo_Cohort_Revenue_Jan2025_to_Mar2026",
    }

    for key, frame in frames.items():
        stem = name_map[key]
        frame.to_csv(csv_dir / f"{stem}.csv", index=False)
        with pd.ExcelWriter(xlsx_dir / f"{stem}.xlsx", engine="xlsxwriter") as writer:
            frame.to_excel(writer, index=False, sheet_name="Data")


def main() -> None:
    frames = build_demo_pack()
    write_outputs(frames)
    for key, frame in frames.items():
        print(f"{key}: {len(frame):,} rows")


if __name__ == "__main__":
    main()
