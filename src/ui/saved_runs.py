from pathlib import Path

import pandas as pd
import streamlit as st

from src.storage.run_store import list_saved_runs, load_run_bundle


def render_saved_runs(base_dir: Path) -> None:
    st.title("Saved Runs")

    if not st.session_state.project_id:
        st.info("Open a project first.")
        return

    runs = list_saved_runs(base_dir, st.session_state.project_id)
    if not runs:
        st.info("No saved runs for this project yet.")
        return

    selected = st.selectbox(
        "Choose a saved run",
        options=runs,
        format_func=lambda r: f"{r['run_id']} | {r['created_at']}",
    )

    if st.button("Load Saved Run"):
        bundle = load_run_bundle(base_dir, st.session_state.project_id, selected["run_id"])
        st.session_state.results_bundle = bundle
        st.success("Run loaded into session.")

    st.subheader("Available runs")
    st.dataframe(pd.DataFrame(runs), use_container_width=True)
