import streamlit as st

from src.storage.project_store import create_project, list_projects, load_project_metadata


def render_home() -> None:
    st.title("Forecast App")
    st.write("Build and compare revenue forecasts across multiple model families.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Start New Project")
        project_name = st.text_input("Project name", value="Revenue Forecast Project")
        if st.button("Create Project", type="primary"):
            metadata = create_project(project_name)
            st.session_state.project_id = metadata["project_id"]
            st.session_state.project_name = metadata["project_name"]
            st.session_state.project_dir = metadata["project_dir"]
            st.success(f"Created project: {metadata['project_name']}")

    with col2:
        st.subheader("Open Existing Project")
        projects = list_projects()
        if projects:
            selected = st.selectbox(
                "Select project",
                options=projects,
                format_func=lambda x: f"{x['project_name']} ({x['project_id']})",
            )
            if st.button("Open Project"):
                metadata = load_project_metadata(selected["project_id"])
                st.session_state.project_id = metadata["project_id"]
                st.session_state.project_name = metadata["project_name"]
                st.session_state.project_dir = metadata["project_dir"]
                st.success(f"Opened project: {metadata['project_name']}")
        else:
            st.info("No saved projects yet.")

    st.divider()
    st.write("Current project")
    st.json(
        {
            "project_id": st.session_state.project_id,
            "project_name": st.session_state.project_name,
            "project_dir": st.session_state.project_dir,
        }
    )
