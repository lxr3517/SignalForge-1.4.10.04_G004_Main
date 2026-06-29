import pandas as pd

from app.utils.web_helpers import _build_cohort_revenue_visuals


def test_cohort_roas_and_payback_are_distinct_metrics(tmp_path):
    cohort_path = tmp_path / "cohort_revenue.csv"
    cost_path = tmp_path / "cost.csv"

    pd.DataFrame(
        [
            {"lead_month": "2026-01-01", "transaction_month": "2026-01-01", "revenue": 80},
            {"lead_month": "2026-01-01", "transaction_month": "2026-02-01", "revenue": 70},
            {"lead_month": "2026-02-01", "transaction_month": "2026-02-01", "revenue": 60},
        ]
    ).to_csv(cohort_path, index=False)
    pd.DataFrame(
        [
            {"date": "2026-01-01", "cost": 100},
            {"date": "2026-02-01", "cost": 100},
        ]
    ).to_csv(cost_path, index=False)

    upload_meta = {
        "files": {
            "cohort_revenue": {"path": str(cohort_path), "filename": cohort_path.name},
            "cost": {"path": str(cost_path), "filename": cost_path.name},
        }
    }
    mapping = {
        "cohort_revenue": {
            "file_key": "cohort_revenue",
            "lead_month_column": "lead_month",
            "transaction_month_column": "transaction_month",
            "value_column": "revenue",
        },
        "cost": {
            "file_key": "cost",
            "date_column": "date",
            "value_column": "cost",
        },
    }

    visuals = _build_cohort_revenue_visuals(upload_meta, mapping)
    by_month = {row["lead_month"]: row for row in visuals["roas_table"]}

    assert by_month["2026-Jan"]["roas"] == 1.5
    assert by_month["2026-Jan"]["roas_pct"] == 150.0
    assert by_month["2026-Jan"]["payback_pct"] == 100.0
    assert by_month["2026-Feb"]["roas"] == 0.6
    assert by_month["2026-Feb"]["roas_pct"] == 60.0
    assert by_month["2026-Feb"]["payback_pct"] == 60.0


def test_payback_summary_uses_latest_12_lead_months(tmp_path):
    cohort_path = tmp_path / "cohort_revenue.csv"
    cost_path = tmp_path / "cost.csv"

    cohort_rows = []
    cost_rows = []
    for idx, month in enumerate(pd.date_range("2024-01-01", periods=15, freq="MS")):
        revenue = 1000 if idx == 0 else 80
        cost = 100 if idx == 0 else 100
        cohort_rows.append({
            "lead_month": month.strftime("%Y-%m-%d"),
            "transaction_month": month.strftime("%Y-%m-%d"),
            "revenue": revenue,
        })
        cost_rows.append({"date": month.strftime("%Y-%m-%d"), "cost": cost})

    pd.DataFrame(cohort_rows).to_csv(cohort_path, index=False)
    pd.DataFrame(cost_rows).to_csv(cost_path, index=False)

    upload_meta = {
        "files": {
            "cohort_revenue": {"path": str(cohort_path), "filename": cohort_path.name},
            "cost": {"path": str(cost_path), "filename": cost_path.name},
        }
    }
    mapping = {
        "cohort_revenue": {
            "file_key": "cohort_revenue",
            "lead_month_column": "lead_month",
            "transaction_month_column": "transaction_month",
            "value_column": "revenue",
        },
        "cost": {
            "file_key": "cost",
            "date_column": "date",
            "value_column": "cost",
        },
    }

    visuals = _build_cohort_revenue_visuals(upload_meta, mapping)
    summary_months = {row["lead_month"] for row in visuals["payback_summary"]}

    assert "2024-Jan" not in summary_months


def test_platform_rollup_does_not_scale_missing_recent_roas(tmp_path):
    cohort_path = tmp_path / "cohort_revenue.csv"
    cost_path = tmp_path / "cost.csv"

    cohort_rows = []
    for idx, month in enumerate(pd.date_range("2024-01-01", periods=14, freq="MS")):
        cohort_rows.append({
            "lead_month": month.strftime("%Y-%m-%d"),
            "transaction_month": month.strftime("%Y-%m-%d"),
            "platform": "Meta",
            "revenue": 1000 if idx >= 2 else 100,
        })
    pd.DataFrame(cohort_rows).to_csv(cohort_path, index=False)

    pd.DataFrame([
        {"date": "2024-01-01", "platform": "Meta", "cost": 100},
        {"date": "2024-02-01", "platform": "Meta", "cost": 100},
    ]).to_csv(cost_path, index=False)

    upload_meta = {
        "files": {
            "cohort_revenue": {"path": str(cohort_path), "filename": cohort_path.name},
            "cost": {"path": str(cost_path), "filename": cost_path.name},
        }
    }
    mapping = {
        "cohort_revenue": {
            "file_key": "cohort_revenue",
            "lead_month_column": "lead_month",
            "transaction_month_column": "transaction_month",
            "value_column": "revenue",
            "category_column": "platform",
        },
        "cost": {
            "file_key": "cost",
            "date_column": "date",
            "value_column": "cost",
            "category_column": "platform",
        },
    }

    visuals = _build_cohort_revenue_visuals(upload_meta, mapping)
    meta = next(row for row in visuals["platform_table"] if "Meta" in row["cohort_category"])

    assert meta["recent_roas"] is None
    assert meta["action"] != "Scale"
    assert "Verify" in meta["action"]
