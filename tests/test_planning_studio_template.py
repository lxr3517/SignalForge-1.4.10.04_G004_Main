from pathlib import Path


def test_planning_studio_uses_reconciled_baseline_math():
    template = Path("app/templates/results.html").read_text(encoding="utf-8")

    assert "currentRoas: safeDivide(currentRevenue, currentBudget)" in template
    assert "currentRpl: safeDivide(currentRevenue, currentLeads)" in template
    assert "const latestCompleteRow" in template
    assert "const revenue = locked ? planningNum(ctx.currentRevenue)" in template
    assert "const roas = locked ? planningNum(ctx.currentRoas)" in template
    assert "Marginal ROAS on extra spend" in template
