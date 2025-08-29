
import streamlit as st
from orchestrator import orchestrate
from gui.components.sidebar import (
    render_sidebar_branding,
    render_sidebar_tabs,
    render_sidebar_file_upload,
    render_workflow_blocks,
)

# --- Handle new workplan tab creation ---
if st.session_state.pop("create_new_tab", False):
    workplan_tabs = st.session_state.setdefault("workplan_tabs", [])
    workplan_configs = st.session_state.setdefault("workplan_configs", {})
    # Generate a unique tab name
    base_name = "Workplan"
    idx = 1
    while f"{base_name} {idx}" in workplan_tabs:
        idx += 1
    new_tab = f"{base_name} {idx}"
    workplan_tabs.append(new_tab)
    st.session_state["active_tab"] = new_tab
    # Optionally, initialize config for the new tab here if not already handled

# --- Sidebar: branding, tab controls, file upload ---
render_sidebar_branding()
render_sidebar_tabs(
    tab_cap=5,
    workplan_tabs=st.session_state.get("workplan_tabs", []),
    active_tab=st.session_state.get("active_tab"),
)
render_sidebar_file_upload(st.session_state.get("active_tab"))


# --- Main page: Only show workplan config and YAML if a workplan tab is active ---
active_tab = st.session_state.get("active_tab")
workplan_tabs = st.session_state.get("workplan_tabs", [])
workplan_configs = st.session_state.setdefault("workplan_configs", {})

if active_tab:
    # Ensure each tab has its own config in session state
    if active_tab not in workplan_configs:
        # Initialize with default config structure
        workplan_configs[active_tab] = {
            "workflow_steps": [
                {"component": "data_block", "configs": []},
                {"component": "judge_block", "configs": []},
                {"component": "analytics_block", "configs": []}
            ],
            "workplan_active": True,
            "study": {},
        }
    # Set tab_state for workflow rendering
    st.session_state["tab_state"] = workplan_configs[active_tab]

    # Layout: sidebar (already rendered), main workflow, YAML preview
    col_main, col_yaml = st.columns([2, 1], gap="large")
    with col_main:
        render_workflow_blocks()
        st.markdown("---")
        st.header(f"Run Study: {active_tab}")
        # Run Study controls (per tab)
        config_path = st.text_input(f"Config YAML Path for {active_tab}", value="configs/analyst_run.yaml", key=f"config_path_{active_tab}")
        progress_bar = st.progress(0, text="Ready.")
        progress_text = st.empty()
        def gui_progress_callback(current, total, message):
            percent = int((current / total) * 100) if total else 0
            progress_bar.progress(percent / 100.0, text=message)
            progress_text.text(message)
        if st.button(f"Run Study for {active_tab}", key=f"run_study_btn_{active_tab}"):
            result = orchestrate(config_path, progress_callback=gui_progress_callback)
            progress_bar.progress(1.0, text="Complete!")
            progress_text.text("Complete!")
            st.success("Study run complete.")
            st.write(result)
    with col_yaml:
        import yaml
        workflow_yaml = yaml.safe_dump(workplan_configs[active_tab], sort_keys=False, allow_unicode=True)
        tab_idx = workplan_tabs.index(active_tab) if active_tab in workplan_tabs else 0
        st.markdown(f"#### Live YAML for {active_tab}")
        st.text_area(
            "",
            value=workflow_yaml,
            height=500,
            key=f"workflow_text_area_{active_tab}_{tab_idx}"
        )
else:
    st.info("Select or create a workplan tab to begin.")
