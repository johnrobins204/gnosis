# Minimal Streamlit analytics GUI scaffold
import streamlit as st
import yaml
import json

# --- Multi-workplan tab management ---
if "workplan_tabs" not in st.session_state:
    st.session_state["workplan_tabs"] = []  # List of tab names/IDs
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = None
if "tab_states" not in st.session_state:
    st.session_state["tab_states"] = {}  # Dict: tab_id -> {study, workflow_steps, workplan_active}

WORKPLAN_TAB_CAP = 5  # Default cap for open workplan tabs
# TODO (backlog): Create an application configuration file to allow power user editing of settings like WORKPLAN_TAB_CAP
def create_new_tab(tab_name=None, study_obj=None):
	# Enforce tab cap
	if len(st.session_state["workplan_tabs"]) >= WORKPLAN_TAB_CAP:
		st.sidebar.warning(f"Tab cap of {WORKPLAN_TAB_CAP} reached. Close a tab before creating a new one.")
		return
	# Generate a unique tab name if not provided
	if not tab_name:
		tab_name = f"Workplan {len(st.session_state['workplan_tabs'])+1}"
	if tab_name in st.session_state["workplan_tabs"]:
		return  # Already exists
	st.session_state["workplan_tabs"].append(tab_name)
	st.session_state["active_tab"] = tab_name
	# Initialize tab state
	st.session_state["tab_states"][tab_name] = {
		"study": study_obj if study_obj else {"name": "My Study", "description": "", "workflow": {"steps": []}},
		"workflow_steps": study_obj.get("workflow", {}).get("steps", []) if study_obj else [],
		"workplan_active": True
	}

def switch_tab(tab_name):
    st.session_state["active_tab"] = tab_name

def deactivate_workplan_builder(tab_name):
    st.session_state["tab_states"][tab_name]["workplan_active"] = False

def activate_workplan_builder(tab_name):
    st.session_state["tab_states"][tab_name]["workplan_active"] = True

# Helper: get current tab state

def get_tab_state():
    tab = st.session_state.get("active_tab")
    if not tab:
        return None
    return st.session_state["tab_states"].get(tab)

# --- Workplan builder state management ---


# --- Sidebar: Configs Pane ---
st.sidebar.title("Configs Pane")


# --- Sidebar: Work Plan Controls ---
st.sidebar.header("Workplan builder")

# New Workplan button (always creates a new tab and switches to it)
if st.sidebar.button("New Workplan"):
	if len(st.session_state["workplan_tabs"]) >= WORKPLAN_TAB_CAP:
		st.sidebar.warning(f"Tab cap of {WORKPLAN_TAB_CAP} reached. No new tab created.")
	else:
		create_new_tab()

# --- Tab selector UI ---
if st.session_state["workplan_tabs"]:
	selected_tab = st.sidebar.selectbox(
		"Open Workplans",
		st.session_state["workplan_tabs"],
		index=st.session_state["workplan_tabs"].index(st.session_state["active_tab"]) if st.session_state["active_tab"] in st.session_state["workplan_tabs"] else 0,
		key="workplan_tab_selector"
	)
	if selected_tab != st.session_state["active_tab"]:
		st.session_state["active_tab"] = selected_tab

# Open Work Plan button and file uploader
uploaded_plan = st.sidebar.file_uploader("Open Work Plan ...", type=["yaml", "yml", "json"], label_visibility="visible")
if uploaded_plan is not None:
	import io
	file_content = uploaded_plan.read()
	try:
		if uploaded_plan.name.endswith((".yaml", ".yml")):
			loaded = yaml.safe_load(io.BytesIO(file_content))
		else:
			loaded = json.load(io.BytesIO(file_content))
		# Accept either {"study": ...} or just the study object
		if "study" in loaded:
			study_obj = loaded["study"]
		else:
			study_obj = loaded
		# Only create a new tab if not already open
		if not st.session_state["active_tab"]:
			create_new_tab(tab_name=study_obj.get("name", None), study_obj=study_obj)
			st.sidebar.success("Work plan loaded!")
		else:
			st.sidebar.warning("A workplan tab is already open. Close it before opening another.")
	except Exception as e:
		st.sidebar.error(f"Failed to load work plan: {e}")

# Spacer bar between Open Work Plan and config selectbox
st.sidebar.markdown('---')

# Navigation dropdown at the top of the sidebar
config_options = [
    "Data Source",
    "Inference Config",
    "Prompt Config",
    "Judge Config",
    "Analytics Config"
]
active_tab = st.session_state.get("active_tab")
tab_states = st.session_state.get("tab_states", {})
tab_state = tab_states.get(active_tab) if active_tab else None
if tab_state and tab_state.get("workplan_active"):
    selected_config = st.sidebar.selectbox("Workplan Elements", config_options, disabled=False)
else:
    selected_config = st.sidebar.selectbox("Workplan Elements", config_options, disabled=True)

# Always define show_configs, steps, and first_step_allowed before any config form block
show_configs = tab_state.get("workplan_active", False) if tab_state else False
steps = tab_state["workflow_steps"] if tab_state and tab_state.get("workflow_steps") is not None else []
if not steps:
    first_step_allowed = selected_config in ["Inference Config", "Data Source"]
else:
    first_step_allowed = True

# Data Source Config Form
if show_configs and selected_config == "Data Source":
    if tab_state is None:
        st.sidebar.error("No active workplan tab. Please create or select a workplan tab first.")
    elif not steps or steps[0]["component"] != "inference":
        st.sidebar.subheader("Data Source Configuration")
        with st.sidebar.form("data_source_form"):
            dataset_path = st.text_input("Dataset path or URI", value="")
            dataset_desc = st.text_area("Description / Metadata", value="", height=60)
            add_data_source = st.form_submit_button("Add to plan", disabled=not first_step_allowed)
            if add_data_source:
                step_cfg = {
                    "component": "data_source",
                    "config": {
                        "dataset_path": dataset_path,
                        "description": dataset_desc
                    }
                }
                if tab_state.get("workflow_steps") is not None:
                    tab_state["workflow_steps"].insert(0, step_cfg)  # Always insert at the start
                    st.rerun()
                else:
                    st.sidebar.error("Internal error: workflow_steps not initialized.")
    else:
        st.sidebar.info("Cannot add a Data Source after an Inference Config. The first step must be either a Data Source or an Inference Config.")

# Inference Config Form
if show_configs and selected_config == "Inference Config":
    if tab_state is None:
        st.sidebar.error("No active workplan tab. Please create or select a workplan tab first.")
    elif not steps or steps[0]["component"] != "data_source":
        st.sidebar.subheader("Inference Configuration")
        # Guardrail: Only one inference step per workplan
        has_inference = any(s.get("component") == "inference" for s in tab_state["workflow_steps"]) if tab_state else False
        with st.sidebar.form("inference_form"):
            default_model = st.text_input("Model", value="google:default")
            input_csv = st.text_input("Question Bank (CSV/YAML/JSON)", value="data/prompts.csv")
            iterations = st.number_input("Iterations per question", min_value=1, max_value=100, value=1)
            temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.01)
            token_limit = st.number_input("Token limit", min_value=1, max_value=4096, value=512)
            top_p = st.number_input("Top-p (nucleus sampling)", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
            top_k = st.number_input("Top-k", min_value=0, max_value=100, value=0)
            seed = st.number_input("Seed", min_value=0, max_value=2**32-1, value=42)
            stop_sequences = st.text_area("Stop sequences (comma-separated)", value="")
            batch_size = st.number_input("Batch size", min_value=1, max_value=1024, value=1)
            output_format = st.selectbox("Output format", ["csv", "json", "parquet", "postgres"], index=0)
            output_csv = None
            if output_format != "postgres":
                output_csv = st.text_input("Output file name", value="artifacts/inference_raw.csv", disabled=False)
            log_level = st.selectbox("Log level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
            add_to_plan = st.form_submit_button("Add to plan", disabled=has_inference)
            if add_to_plan:
                step_cfg = {
                    "component": "inference",
                    "config": {
                        "default_model": default_model,
                        "input_csv": input_csv,
                        "iterations": iterations,
                        "temperature": temperature,
                        "token_limit": token_limit,
                        "top_p": top_p,
                        "top_k": top_k,
                        "seed": seed,
                        "stop_sequences": [s.strip() for s in stop_sequences.split(",") if s.strip()],
                        "batch_size": batch_size,
                        "output_format": output_format,
                        "output_csv": output_csv if output_format != 'postgres' else None,
                        "log_level": log_level
                    }
                }
                tab_state["workflow_steps"].append(step_cfg)
                st.rerun()
            if has_inference:
                st.sidebar.info("Only one inference step is allowed per workplan.")
    else:
        st.sidebar.info("Cannot add an Inference Config after a Data Source. The first step must be either a Data Source or an Inference Config.")

# Prompt Config Form
if show_configs and selected_config == "Prompt Config":
    if tab_state is None:
        st.sidebar.error("No active workplan tab. Please create or select a workplan tab first.")
    elif not steps or steps[0]["component"] in ["inference", "data_source"]:
        st.sidebar.subheader("Prompt Configuration")
        # Get defaults from the main inference step if present
        inference_step = next((s for s in tab_state["workflow_steps"] if s.get("component") == "inference"), None) if tab_state else None
        inf_cfg = inference_step["config"] if inference_step else {}
        with st.sidebar.form("prompt_form"):
            default_model = st.text_input("Model (override)", value=inf_cfg.get("default_model", "google:default"))
            temperature = st.number_input("Temperature (override)", min_value=0.0, max_value=2.0, value=float(inf_cfg.get("temperature", 1.0)), step=0.01)
            token_limit = st.number_input("Token limit (override)", min_value=1, max_value=4096, value=int(inf_cfg.get("token_limit", 512)))
            top_p = st.number_input("Top-p (override)", min_value=0.0, max_value=1.0, value=float(inf_cfg.get("top_p", 1.0)), step=0.01)
            top_k = st.number_input("Top-k (override)", min_value=0, max_value=100, value=int(inf_cfg.get("top_k", 0)))
            prompt_injection = st.text_area("Prompt injection", value="", height=100)
            json_injection = st.text_area("JSON injection", value="", height=100)
            add_prompt = st.form_submit_button("Add to plan")
            if add_prompt:
                step_cfg = {
                    "component": "prompt",
                    "config": {
                        "default_model": default_model,
                        "temperature": temperature,
                        "token_limit": token_limit,
                        "top_p": top_p,
                        "top_k": top_k,
                        "prompt_injection": prompt_injection,
                        "json_injection": json_injection
                    }
                }
                tab_state["workflow_steps"].append(step_cfg)
                st.rerun()
    else:
        st.sidebar.info("You must start your workplan with an Inference Config or Data Source.")

# Judge Config Form
if show_configs and selected_config == "Judge Config":
    if tab_state is None:
        st.sidebar.error("No active workplan tab. Please create or select a workplan tab first.")
    elif not steps or steps[0]["component"] in ["inference", "data_source"]:
        import os, json
        st.sidebar.subheader("Judge Configuration")
        # Get defaults from the main inference step if present
        inference_step = next((s for s in tab_state["workflow_steps"] if s.get("component") == "inference"), None) if tab_state else None
        inf_cfg = inference_step["config"] if inference_step else {}
        # --- Judge template selection ---
        template_dir = os.path.join(os.path.dirname(__file__), '../../configs/injection_templates')
        template_files = [f for f in os.listdir(template_dir) if f.startswith('judge_') and f.endswith('.json')]
        template_files.sort()
        template_labels = []
        template_jsons = []
        for f in template_files:
            with open(os.path.join(template_dir, f), 'r') as tf:
                j = json.load(tf)
                # Use the first key in 'evaluationCriteria' as the display name
                if isinstance(j.get('evaluationCriteria'), dict) and len(j['evaluationCriteria']) > 0:
                    label = list(j['evaluationCriteria'].keys())[0]
                else:
                    label = f.replace('judge_', '').replace('.json', '').replace('_', ' ').title()
                template_labels.append(label)
                template_jsons.append(j)
        selected_idx = st.sidebar.selectbox("Judge Template", template_labels, index=0 if template_labels else None, key="judge_template_select") if template_labels else None
        selected_template_json = ""
        if selected_idx is not None and template_labels:
            selected_file = template_files[template_labels.index(selected_idx)]
            with open(os.path.join(template_dir, selected_file), 'r') as f:
                selected_template_json = f.read()
            st.sidebar.text_area("Template Preview (JSON)", value=selected_template_json, height=150, key="judge_template_preview")
        with st.sidebar.form("judge_form"):
            default_model = st.text_input("Model (override)", value=inf_cfg.get("default_model", "google:default"))
            temperature = st.number_input("Temperature (override)", min_value=0.0, max_value=2.0, value=float(inf_cfg.get("temperature", 1.0)), step=0.01)
            token_limit = st.number_input("Token limit (override)", min_value=1, max_value=4096, value=int(inf_cfg.get("token_limit", 512)))
            top_p = st.number_input("Top-p (override)", min_value=0.0, max_value=1.0, value=float(inf_cfg.get("top_p", 1.0)), step=0.01)
            top_k = st.number_input("Top-k (override)", min_value=0, max_value=100, value=int(inf_cfg.get("top_k", 0)))
            judge_instructions = st.text_area("Judge instructions/template", value=selected_template_json, height=100)
            add_judge = st.form_submit_button("Add to plan")
            if add_judge:
                step_cfg = {
                    "component": "judge",
                    "config": {
                        "default_model": default_model,
                        "temperature": temperature,
                        "token_limit": token_limit,
                        "top_p": top_p,
                        "top_k": top_k,
                        "judge_instructions": judge_instructions
                    }
                }
                tab_state["workflow_steps"].append(step_cfg)
                st.rerun()
    else:
        st.sidebar.info("You must start your workplan with an Inference Config or Data Source.")

# Analytics Config Form
if show_configs and selected_config == "Analytics Config":
    if tab_state is None:
        st.sidebar.error("No active workplan tab. Please create or select a workplan tab first.")
    elif not steps or steps[0]["component"] in ["inference", "data_source"]:
        st.sidebar.subheader("Analytics Configuration")
        st.sidebar.info("Analytics config UI coming soon.")
    else:
        st.sidebar.info("You must start your workplan with an Inference Config or Data Source.")


# --- Persistent Workflow Plan (always at bottom) ---
# --- Main pane content ---
active_tab = st.session_state.get("active_tab")
tab_states = st.session_state.get("tab_states", {})
if not active_tab or active_tab not in tab_states or not tab_states[active_tab].get("workplan_active"):
    st.markdown("# Hello World")
else:
    tab_state = tab_states[active_tab]
    st.markdown("---")
    st.subheader(f"Workflow Plan: {active_tab}")
    # Study-level metadata
    study = tab_state["study"]
    # Step 0: Workflow-level configs
    with st.expander("Step 0: Workflow-level Configs", expanded=True):
        study["name"] = st.text_input("Study Name", value=study.get("name", "My Study"), key=f"workflow_study_name_{active_tab}")
        study["description"] = st.text_area("Study Description", value=study.get("description", ""), height=50, key=f"workflow_study_desc_{active_tab}")
    # Steps
    study["workflow"]["steps"] = tab_state["workflow_steps"]
    # Render each step in its own box
    for i, step in enumerate(study["workflow"]["steps"]):
        with st.expander(f"Step {i+1}: {step.get('component', 'Unknown')}", expanded=True):
            st.code(yaml.safe_dump(step, sort_keys=False, allow_unicode=True), language="yaml")
    # Show full hierarchical YAML
    workflow_yaml = yaml.safe_dump({"study": study}, sort_keys=False, allow_unicode=True)
    st.text_area("Full Workflow YAML", value=workflow_yaml, height=250, key=f"workflow_text_area_{active_tab}")



# --- Run Study Button and Progress Bar ---
from src.orchestrator import orchestrate

st.markdown("---")
st.header("Run Study")

# Assume config_path is provided or constructed from the current workflow
config_path = st.text_input("Config YAML Path", value="configs/analyst_run.yaml")

progress_bar = st.progress(0, text="Ready.")
progress_text = st.empty()

def gui_progress_callback(current, total, message):
    percent = int((current / total) * 100) if total else 0
    progress_bar.progress(percent / 100.0, text=message)
    progress_text.text(message)

if st.button("Run Study"):
    result = orchestrate(config_path, progress_callback=gui_progress_callback)
    progress_bar.progress(1.0, text="Complete!")
    progress_text.text("Complete!")
    st.success("Study run complete.")
    st.write(result)

# --- Ensure workplan_active is in sync with tab state ---
def sync_workplan_active():
    tabs = st.session_state.get("workplan_tabs", [])
    active_tab = st.session_state.get("active_tab")
    tab_states = st.session_state.get("tab_states", {})
    if tabs and active_tab in tab_states:
        tab_states[active_tab]["workplan_active"] = True
    else:
        st.session_state["workplan_active"] = False

# Call this after any tab state change
sync_workplan_active()
