from pathlib import Path

import streamlit as st

from src.evaluation.ranking import choose_best_and_blend
from src.explainability.narrative import build_narrative
from src.models.ensemble import make_scenario_frame
from src.models.lightgbm_model import run_lightgbm_forecast
from src.models.prophet_model import run_prophet_forecast
from src.models.statsforecast_models import run_statsforecast_forecast
from src.models.baselines import run_baseline_forecasts
from src.storage.run_store import save_run_bundle


def render_forecast_setup(base_dir: Path) -> None:
    st.title("Forecast Setup")

    if st.session_state.normalized_df is None:
        st.warning("Load and map a dataset first.")
        return

    targets = ["revenue", "roas", "leads", "spend", "conversions", "custom"]
    col1, col2, col3 = st.columns(3)

    with col1:
        target = st.selectbox("Forecast target", targets, index=0)
        frequency = st.selectbox("Frequency", ["D", "W", "M"], index=0)
    with col2:
        horizon = st.selectbox("Forecast horizon", ["7D", "30D", "90D", "12M"], index=0)
        compare_mode = st.selectbox(
            "Compare mode",
            ["blend_vs_best", "best_only"],
            index=0,
        )
    with col3:
        evaluation_preference = st.selectbox(
            "Evaluation preference",
            ["balanced", "mae", "rmse", "smape", "bias", "recent_accuracy"],
            index=0,
        )
        output_mode = st.selectbox(
            "Output mode",
            ["scenario_bands", "confidence_intervals", "single_line"],
            index=0,
        )

    explainability = st.checkbox("Enable explainability", value=True)
    tuning_depth = st.selectbox("Automatic tuning depth", ["fast", "balanced", "aggressive"], index=1)

    if st.button("Run Forecast", type="primary"):
        config = {
            "target": target,
            "frequency": frequency,
            "horizon": horizon,
            "compare_mode": compare_mode,
            "evaluation_preference": evaluation_preference,
            "output_mode": output_mode,
            "explainability": explainability,
            "tuning_depth": tuning_depth,
        }
        st.session_state.forecast_config.update(config)
        df = st.session_state.normalized_df.copy()

        baseline_bundle = run_baseline_forecasts(df, config)
        stats_bundle = run_statsforecast_forecast(df, config)
        prophet_bundle = run_prophet_forecast(df, config)
        lgbm_bundle = run_lightgbm_forecast(df, config)

        all_metrics = (
            baseline_bundle["metrics"]
            + stats_bundle["metrics"]
            + prophet_bundle["metrics"]
            + lgbm_bundle["metrics"]
        )
        all_forecasts = (
            baseline_bundle["forecasts"]
            + stats_bundle["forecasts"]
            + prophet_bundle["forecasts"]
            + lgbm_bundle["forecasts"]
        )

        decision = choose_best_and_blend(all_metrics, all_forecasts, evaluation_preference)
        scenarios = make_scenario_frame(decision["blended_forecast"])
        narrative = build_narrative(df, decision, config)

        bundle = {
            "config": config,
            "all_metrics": all_metrics,
            "all_forecasts": all_forecasts,
            "decision": decision,
            "scenarios": scenarios,
            "narrative": narrative,
        }
        st.session_state.results_bundle = bundle
        save_run_bundle(base_dir, st.session_state.project_id, bundle)
        st.success("Forecast completed. Open the Results page.")
