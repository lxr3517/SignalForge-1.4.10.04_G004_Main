import streamlit as st

from src.preprocessing.adjustments import apply_manual_adjustments
from src.preprocessing.cleaning import generate_quality_report


def render_data_quality() -> None:
    st.title("Data Quality & Adjustments")

    if st.session_state.normalized_df is None:
        st.warning("Map a revenue dataset first on the Data Setup page.")
        return

    df = st.session_state.normalized_df.copy()
    report = generate_quality_report(df)
    st.session_state.quality_report = report

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Auto-detected issues")
        st.json(report)
    with col2:
        st.subheader("Manual adjustments")
        exclude_start = st.date_input("Exclude start date", value=None)
        exclude_end = st.date_input("Exclude end date", value=None)
        mark_outage = st.checkbox("Mark selected range as outage")
        mark_promo = st.checkbox("Mark selected range as promo period")
        use_holidays = st.checkbox("Enable holiday effects", value=False)
        lagged_logic = st.checkbox("Enable lagged revenue logic", value=True)
        partial_mode = st.selectbox(
            "Partial period handling",
            ["flag_only", "exclude", "keep", "prorate"],
            index=0,
        )

    if st.button("Apply Manual Adjustments", type="primary"):
        adjusted = apply_manual_adjustments(
            df=df,
            exclude_start=exclude_start,
            exclude_end=exclude_end,
            mark_outage=mark_outage,
            mark_promo=mark_promo,
            use_holidays=use_holidays,
            lagged_logic=lagged_logic,
            partial_mode=partial_mode,
        )
        st.session_state.normalized_df = adjusted
        st.session_state.forecast_config["manual_adjustments"] = {
            "exclude_start": str(exclude_start) if exclude_start else None,
            "exclude_end": str(exclude_end) if exclude_end else None,
            "mark_outage": mark_outage,
            "mark_promo": mark_promo,
            "use_holidays": use_holidays,
            "lagged_logic": lagged_logic,
            "partial_mode": partial_mode,
        }
        st.success("Adjustments applied.")
        st.dataframe(adjusted.head(20), use_container_width=True)
