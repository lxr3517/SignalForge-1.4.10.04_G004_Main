from __future__ import annotations

import streamlit as st

from src.ingestion.loaders import load_uploaded_file
from src.ingestion.schema import apply_column_mapping, infer_candidate_columns
from src.storage.project_store import save_uploaded_dataset


def render_data_setup() -> None:
    st.title("Data Setup")

    if not st.session_state.project_id:
        st.warning("Create or open a project first from the Home page.")
        return

    config = st.session_state.app_config

    col1, col2, col3 = st.columns(3)
    with col1:
        scope = st.selectbox(
            "Forecast scope",
            ["company_total", "by_affiliate", "by_platform", "custom_category"],
            index=0,
        )
    with col2:
        source_type = st.selectbox(
            "Source type",
            ["csv", "excel", "parquet"],
            index=0,
        )
    with col3:
        history_years = st.number_input(
            "Years of history",
            min_value=1,
            max_value=20,
            value=config["app"]["default_history_years"],
            step=1,
        )

    st.session_state.forecast_config["scope"] = scope
    st.session_state.forecast_config["source_type"] = source_type
    st.session_state.forecast_config["history_years"] = history_years

    st.subheader("Core revenue file")
    revenue_file = st.file_uploader(
        "Upload revenue data",
        type=["csv", "xlsx", "xls", "parquet"],
        key="revenue_file",
    )

    optional_inputs = {
        "leads": st.checkbox("Add leads data"),
        "spend": st.checkbox("Add spend data"),
        "traffic": st.checkbox("Add traffic data"),
        "conversions": st.checkbox("Add conversions data"),
        "events": st.checkbox("Add events/holidays"),
        "outages": st.checkbox("Add outages/login issues"),
        "custom_regressors": st.checkbox("Add custom regressors"),
    }
    st.session_state.forecast_config["optional_inputs"] = optional_inputs

    uploaded_frames = {}
    if revenue_file is not None:
        revenue_df = load_uploaded_file(revenue_file)
        uploaded_frames["revenue"] = revenue_df
        save_uploaded_dataset(st.session_state.project_id, "revenue", revenue_file, revenue_df)
        st.write("Preview")
        st.dataframe(revenue_df.head(20), use_container_width=True)

        candidates = infer_candidate_columns(revenue_df)
        st.subheader("Column mapping")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            date_col = st.selectbox("Date column", options=candidates, key="date_col")
        with col_b:
            target_col = st.selectbox("Target column", options=candidates, key="target_col")
        with col_c:
            category_options = ["<none>"] + candidates
            category_col = st.selectbox(
                "Source/category column",
                options=category_options,
                key="category_col",
                help="Required for affiliate/platform/custom category forecasting.",
            )

        if st.button("Apply Mapping", type="primary"):
            mapped = apply_column_mapping(
                revenue_df,
                date_col=date_col,
                target_col=target_col,
                category_col=None if category_col == "<none>" else category_col,
            )
            st.session_state.normalized_df = mapped
            st.session_state.forecast_config["column_mapping"] = {
                "date_col": date_col,
                "target_col": target_col,
                "category_col": None if category_col == "<none>" else category_col,
            }
            st.success("Column mapping applied.")
            st.dataframe(mapped.head(20), use_container_width=True)

    for dataset_name, enabled in optional_inputs.items():
        if enabled:
            file = st.file_uploader(
                f"Upload {dataset_name} data",
                type=["csv", "xlsx", "xls", "parquet"],
                key=f"{dataset_name}_file",
            )
            if file is not None:
                df = load_uploaded_file(file)
                uploaded_frames[dataset_name] = df
                save_uploaded_dataset(st.session_state.project_id, dataset_name, file, df)
                st.write(f"{dataset_name.title()} preview")
                st.dataframe(df.head(10), use_container_width=True)

    st.session_state.uploaded_frames = uploaded_frames
