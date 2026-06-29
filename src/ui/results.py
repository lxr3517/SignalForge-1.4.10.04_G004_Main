import pandas as pd
import plotly.express as px
import streamlit as st

from src.exports.export_csv import dataframe_to_csv_bytes
from src.exports.export_excel import bundle_to_excel_bytes


def render_results() -> None:
    st.title("Results")

    bundle = st.session_state.results_bundle
    if bundle is None:
        st.info("No results yet. Run a forecast first.")
        return

    decision = bundle["decision"]
    best_model = decision["best_model"]
    best_forecast = pd.DataFrame(decision["best_forecast"])
    blended_forecast = pd.DataFrame(decision["blended_forecast"])
    scenarios = pd.DataFrame(bundle["scenarios"])

    st.subheader("Best single model")
    st.json(best_model)

    st.subheader("Forecast narrative")
    st.write(bundle["narrative"])

    col1, col2 = st.columns(2)
    with col1:
        st.write("Best single forecast")
        st.dataframe(best_forecast, use_container_width=True)
    with col2:
        st.write("Blended forecast")
        st.dataframe(blended_forecast, use_container_width=True)

    if not scenarios.empty:
        fig = px.line(scenarios, x="ds", y=["conservative", "expected", "aggressive"])
        st.plotly_chart(fig, use_container_width=True)

    ranking_df = pd.DataFrame(decision.get("model_ranking", []))
    metrics_df = ranking_df if not ranking_df.empty else pd.DataFrame(bundle["all_metrics"])
    st.subheader("Model comparison")
    st.dataframe(metrics_df, use_container_width=True)

    st.download_button(
        "Download blended forecast CSV",
        data=dataframe_to_csv_bytes(blended_forecast),
        file_name="blended_forecast.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download full Excel workbook",
        data=bundle_to_excel_bytes(bundle),
        file_name="forecast_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
