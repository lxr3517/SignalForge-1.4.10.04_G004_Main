import json

import pandas as pd

from app.utils.web_helpers import _build_mapped_daily_rollup, _build_mapped_monthly_rollup, _build_results_visuals, _build_scenario2_table


def test_scenario_table_prefers_monthly_planning_baseline():
    rows, cards = _build_scenario2_table(
        {
            "base_revenue": 40000,
            "base_cost": 600,
            "base_leads": 3000,
            "base_rev_per_lead": 13,
            "planning_month_label": "2026-03",
            "planning_monthly_revenue": 1_250_300,
            "planning_monthly_cost": 17_638,
            "planning_monthly_leads": 95_968,
            "planning_monthly_rpl": 13.0283,
        },
        [],
    )

    expected = next(row for row in rows if row["scenario"] == "Expected")

    assert expected["projected_spend"] == 17638.0
    assert expected["projected_leads"] == 95968.0
    assert expected["projected_revenue"] == 1250300.0
    assert round(expected["projected_roas"], 2) == 70.89
    assert cards["period_basis"] == "latest_completed_month"
    assert cards["period_label"] == "2026-03"


def test_planning_studio_cost_uses_raw_mapped_monthly_spend(tmp_path):
    cost_path = tmp_path / "cost.csv"
    leads_path = tmp_path / "leads.csv"

    pd.DataFrame(
        [
            {"lead_day": "2026-03-01", "Platform": "Affiliate", "Cost": 700_000},
            {"lead_day": "2026-03-01", "Platform": "Search", "Cost": 61_852.922},
        ]
    ).to_csv(cost_path, index=False)
    pd.DataFrame(
        [
            {"Lead Date": "2026-03-01", "Platform": "Affiliate", "Count": 80_000},
            {"Lead Date": "2026-03-01", "Platform": "Search", "Count": 15_968},
        ]
    ).to_csv(leads_path, index=False)

    upload_meta = {
        "files": {
            "cost": {"path": str(cost_path), "filename": cost_path.name},
            "leads": {"path": str(leads_path), "filename": leads_path.name},
        }
    }
    mapping = {
        "cost": {"file_key": "cost", "date_column": "lead_day", "value_column": "Cost"},
        "leads": {"file_key": "leads", "date_column": "Lead Date", "value_column": "Count"},
    }

    cost_monthly = _build_mapped_monthly_rollup(upload_meta, mapping, "cost", "cost")
    leads_monthly = _build_mapped_monthly_rollup(upload_meta, mapping, "leads", "leads")
    defaults = {
        "planning_monthly_revenue": 1_250_300,
        "planning_monthly_cost": float(cost_monthly["cost"].iloc[-1]),
        "planning_monthly_leads": float(leads_monthly["leads"].iloc[-1]),
        "planning_monthly_rpl": 1_250_300 / float(leads_monthly["leads"].iloc[-1]),
    }
    rows, _ = _build_scenario2_table(defaults, [])
    expected = next(row for row in rows if row["scenario"] == "Expected")

    assert defaults["planning_monthly_cost"] == 761_852.922
    assert defaults["planning_monthly_leads"] == 95_968
    assert expected["projected_spend"] == 761852.922
    assert round(expected["projected_roas"], 2) == 1.64


def test_lead_diagnostics_average_daily_totals_from_leads_sheet(tmp_path):
    leads_path = tmp_path / "leads.csv"
    pd.DataFrame(
        [
            {"Lead Date": "2026-03-01", "Platform": "Affiliate", "Count": 100},
            {"Lead Date": "2026-03-01", "Platform": "Search", "Count": 50},
            {"Lead Date": "2026-03-02", "Platform": "Affiliate", "Count": 75},
            {"Lead Date": "2026-03-02", "Platform": "Search", "Count": 25},
        ]
    ).to_csv(leads_path, index=False)
    upload_meta = {
        "files": {
            "leads": {"path": str(leads_path), "filename": leads_path.name},
        }
    }
    mapping = {
        "leads": {"file_key": "leads", "date_column": "Lead Date", "value_column": "Count"},
    }

    daily = _build_mapped_daily_rollup(upload_meta, mapping, "leads", "leads")

    assert daily["leads"].tolist() == [150, 100]
    assert daily["leads"].mean() == 125
    assert daily["leads"].mean() * 30 == 3750


def test_diagnostics_ratio_uses_secondary_axis():
    hist = pd.DataFrame(
        {
            "ds": pd.date_range("2026-03-01", periods=35, freq="D"),
            "y": [100 + (i % 5) for i in range(35)],
        }
    )
    result = {
        "diagnostics": {
            "recent_avg_y": 100,
            "forecast_avg_expected": 98,
            "forecast_to_history_ratio": 0.98,
            "avg_leads": 125,
        },
        "settings": {"frequency": "D"},
    }

    visuals = _build_results_visuals(hist, result, {}, {})
    chart = next(c for c in visuals["charts"] if c["id"] == "results-diagnostics")
    spec = json.loads(chart["spec"])

    ratio_trace = next(t for t in spec["data"] if t["name"] == "Forecast / recent ratio")
    assert ratio_trace["yaxis"] == "y2"
    assert "yaxis2" in spec["layout"]
    assert spec["layout"]["yaxis2"]["side"] == "right"
