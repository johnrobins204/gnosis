from gui.components.forms import render_data_source_form, render_judge_block_form, render_analytics_block_form, render_inference_block_form
import yaml

# --- Step Type Registry and Workflow Validation ---
STEP_TYPE_REGISTRY = {
    "data_source": {
        "label": "Data Source",
        "allowed_first": True,
        "allowed_next": ["inference_block", "judge_block", "analytics_block"],
        "single": True,
    },
    "inference_block": {
        "label": "Inference Block",
        "allowed_first": True,
        "allowed_next": ["judge_block", "analytics_block"],
        "single": True,
    },
    "judge_block": {
        "label": "Judge Block",
        "allowed_first": False,
        "allowed_next": ["analytics_block"],
        "single": True,
    },
    "analytics_block": {
        "label": "Analytics Block",
        "allowed_first": False,
        "allowed_next": [],
        "single": True,
    },
}

def get_allowed_next_steps(steps):
    if not steps or len(steps) == 0:
        # Only allow data_source or inference_block as first step
        return [k for k, v in STEP_TYPE_REGISTRY.items() if v["allowed_first"]]
    last = steps[-1]["component"]
    allowed = STEP_TYPE_REGISTRY.get(last, {}).get("allowed_next", [])
    # Prevent multiple data_source or inference_block if single
    for k in list(allowed):
        if STEP_TYPE_REGISTRY[k].get("single") and any(s["component"] == k for s in steps):
            allowed.remove(k)
    return allowed

def render_workflow_blocks():
    # On new workplan, always initialize with data_block, judge_block, analytics_block
    if "tab_state" not in st.session_state or not st.session_state["tab_state"].get("workflow_steps"):
        st.session_state["tab_state"] = {
            "workflow_steps": [
                {"component": "data_block", "configs": []},
                {"component": "judge_block", "configs": []},
                {"component": "analytics_block", "configs": []}
            ],
            "workplan_active": True
        }
    tab_state = st.session_state["tab_state"]
    steps = tab_state["workflow_steps"]
    active_tab = st.session_state.get("active_tab")
    VERBOSE = st.session_state.get("VERBOSE", False)
    if VERBOSE:
        with st.expander("[DEBUG] Workplan State", expanded=True):
            st.write("steps:", steps)
            st.write("tab_state:", tab_state)

    st.header("Workplan Configuration")

    # --- Data Block ---
    st.subheader("Data Block")
    data_block = steps[0]
    # Toggle: dataset upload OR N inference configs
    data_mode = st.radio("Data Block Mode", ["Upload Dataset", "Inference Steps"], key="data_block_mode")
    if data_mode == "Upload Dataset":
        render_data_source_form(tab_state, steps)
    else:
        render_inference_block_form(tab_state, steps)

    # --- Judge Block ---
    st.subheader("Judge Block")
    render_judge_block_form(tab_state, steps)

    # --- Analytics Block ---
    st.subheader("Analytics Block")
    render_analytics_block_form(tab_state, steps)

    # Show full hierarchical YAML
    workflow_yaml = yaml.safe_dump(tab_state, sort_keys=False, allow_unicode=True)
    st.text_area("Full Workflow YAML", value=workflow_yaml, height=250, key=f"workflow_text_area_{active_tab}")
import streamlit as st
import pathlib
import yaml
import json

def render_sidebar_branding():
    import os
    import io
    sidebar_logo_col, sidebar_brand_col = st.sidebar.columns([1, 2])
    with sidebar_logo_col:
        logo_path = os.path.join(os.path.dirname(__file__), "../gnostic_whisper_logo.png")
        import sys
        VERBOSE = any(arg in ("--verbose", "-v") for arg in sys.argv)
        if VERBOSE:
            st.write(f"[DEBUG] logo_path: {logo_path}")
            st.write(f"[DEBUG] os.path.exists(logo_path): {os.path.exists(logo_path)}")
        if os.path.exists(logo_path):
            file_size = os.path.getsize(logo_path)
            if VERBOSE:
                st.write(f"[DEBUG] logo file size: {file_size}")
            with open(logo_path, "rb") as f:
                data = f.read()
                if VERBOSE:
                    st.write(f"[DEBUG] Read {len(data)} bytes from logo file")
                st.image(io.BytesIO(data), width=64)
        else:
            st.error(f"Logo file not found: {logo_path}")
    with sidebar_brand_col:
        st.markdown("<h2 style='color:#89d6ff; font-weight:bold; letter-spacing:2px; margin-bottom:0.2em; text-align:left;'>GNOSTIC_WHISPER</h2>", unsafe_allow_html=True)
        st.markdown("<span style='color:#a0ffff; font-size:0.95em;'>Cosmic 8-Bit Analytics</span>", unsafe_allow_html=True)
    st.sidebar.markdown('<hr style="margin:0.7em 0 1.2em 0; border:1px solid #333;">', unsafe_allow_html=True)

def render_sidebar_tabs(tab_cap, workplan_tabs, active_tab):
    st.sidebar.markdown('<span style="font-size:1.1em; font-weight:bold; color:#89d6ff; letter-spacing:1px;">🗂️ Workplan Builder</span>', unsafe_allow_html=True)
    st.sidebar.markdown("<div style='height:0.5em;'></div>", unsafe_allow_html=True)
    tab_cap_reached = len(workplan_tabs) >= tab_cap
    if tab_cap_reached:
        st.sidebar.button("➕ New Workplan", key="new_workplan_btn", help=f"Tab cap of {tab_cap} reached. Close a tab to create a new one.", disabled=True)
    else:
        new_workplan_clicked = st.sidebar.button("➕ New Workplan", key="new_workplan_btn", help="Create a new workplan tab")
        if new_workplan_clicked:
            st.session_state["create_new_tab"] = True
    if active_tab:
        if st.sidebar.button("❌ Close Active Tab", key="close_tab_btn", help="Close the currently active workplan tab"):
            st.session_state["close_active_tab"] = True
    if workplan_tabs:
        st.sidebar.markdown('<span style="font-size:1em; color:#a0ffff;">🔖 <b>Open Workplans</b></span>', unsafe_allow_html=True)
        selected_tab = st.sidebar.selectbox(
            "Select a workplan tab",  # Non-empty label for accessibility
            workplan_tabs,
            index=workplan_tabs.index(active_tab) if active_tab in workplan_tabs else 0,
            key="workplan_tab_selector",
            label_visibility="collapsed",
            help="Switch between open workplan tabs"
        )
        if selected_tab != active_tab:
            st.session_state["active_tab"] = selected_tab
            st.session_state["show_tab_switched"] = True
        st.sidebar.markdown("<div style='height:0.5em;'></div>", unsafe_allow_html=True)
    if st.session_state.pop("show_tab_switched", False):
        st.sidebar.info("Switched to selected workplan tab.")

def render_sidebar_file_upload(active_tab):
    st.sidebar.markdown('<span style="font-size:1em; color:#a0ffff;">📂 <b>Open Work Plan</b></span>', unsafe_allow_html=True)
    open_disabled = bool(active_tab)
    uploaded_plan = st.sidebar.file_uploader(
        "Upload work plan file",  # Non-empty label for accessibility
        type=["yaml", "yml", "json"],
        label_visibility="collapsed",
        help="Upload a work plan YAML or JSON file",
        disabled=open_disabled
    )
    if open_disabled:
        st.sidebar.info("Close the current workplan tab to open a new one.")
    if uploaded_plan is not None and not open_disabled:
        with st.spinner("Loading workplan..."):
            import io
            file_content = uploaded_plan.read()
            try:
                if uploaded_plan.name.endswith((".yaml", ".yml")):
                    loaded = yaml.safe_load(io.BytesIO(file_content))
                else:
                    loaded = json.load(io.BytesIO(file_content))
                if "study" in loaded:
                    study_obj = loaded["study"]
                else:
                    study_obj = loaded
                if not st.session_state.get("active_tab"):
                    st.session_state["load_workplan"] = study_obj
                    st.session_state["show_workplan_loaded"] = True
                else:
                    st.session_state["show_workplan_open_warning"] = True
            except Exception as e:
                st.session_state["show_workplan_load_error"] = str(e)
    if st.session_state.pop("show_workplan_loaded", False):
        st.sidebar.success("Workplan loaded successfully.")
    if st.session_state.pop("show_workplan_open_warning", False):
        st.sidebar.warning("A workplan tab is already open. Close it before opening another.")
    if "show_workplan_load_error" in st.session_state:
        st.sidebar.error(f"Failed to load work plan: {st.session_state.pop('show_workplan_load_error')}")
    st.sidebar.markdown('<hr style="margin:0.7em 0 1.2em 0; border:1px solid #333;">', unsafe_allow_html=True)
