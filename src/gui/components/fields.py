import streamlit as st
import os
import json
from gui.utils import get_default_output_path

def render_config_field(param, meta, tab_state=None):
    """
    Render a config field based on its type and metadata. Returns the value.
    Handles standardization for input_csv, output_csv/output_path, and type-specific widgets.
    Adds type safety and validation for enums, numerics, and JSON fields.
    """
    # Standardized labels for input/output files
    if param == "input_csv":
        label = "Questions Bank (csv or json)"
        state_key = f"{param}_input"
        uploaded_file = st.file_uploader(label, type=["csv", "json"], key=state_key, help=meta.get("help"))
        if uploaded_file is not None:
            # Only store the file name/path, not the loaded data
            file_path = uploaded_file.name
            st.session_state[state_key + "_path"] = file_path
            from gnosis_io import parse_questions_file
            data, fmt, error = parse_questions_file(uploaded_file)
            if error:
                st.error(error)
            else:
                st.success(f"Loaded questions ({fmt})")
            return file_path
        # If file not uploaded, return previously stored path if available
        if state_key + "_path" in st.session_state:
            return st.session_state[state_key + "_path"]
        if meta.get("required", False):
            st.error(f"{label} is required.")
        return None
    elif param in ("output_path", "output_csv"):
        label = "Output path"
    else:
        label = meta.get("description", param)
    # Accessibility: help text/tooltips
    help_text = meta.get("help")
    required = meta.get("required", False)
    param_type = meta.get("type", "str")
    choices = meta.get("choices")
    min_value = meta.get("min")
    max_value = meta.get("max")
    step = meta.get("step")
    default_val = meta.get("default")
    placeholder = meta.get("placeholder")
    env_var = meta.get("env_var")
    env_val = os.environ.get(env_var) if env_var else None
    state_key = f"{param}_input"

    if param in ("output_path", "output_csv"):
        suggest_flag = f"set_{param}_suggested"
        suggested_val_key = f"{param}_suggested"
        # If the suggest button was clicked, set the flag
        if st.session_state.get(suggest_flag):
            suggested_val = get_default_output_path(tab_state)
            st.session_state[suggested_val_key] = suggested_val
            # Clear the flag after use
            st.session_state[suggest_flag] = False
        # Use suggested value if present, else default
        default = st.session_state.get(suggested_val_key) or default_val or "default.csv"
        # Only set the initial value if not already set
        if state_key not in st.session_state:
            st.session_state[state_key] = default
        # Render the text input
        value = st.text_input(label, value=st.session_state[state_key], key=state_key, placeholder=placeholder or "Enter output file path...", help=help_text)
        # Suggest button sets the flag and triggers rerun
        if st.form_submit_button("Suggest Output Path"):
            st.session_state[suggest_flag] = True
            st.rerun()
        if required and not value:
            st.error(f"{label} is required.")
        return value
    if env_var and env_val is not None:
        st.info(f"Using value from environment variable: {env_var}")
        if state_key not in st.session_state:
            st.session_state[state_key] = env_val
        return env_val
    elif choices:
        if state_key not in st.session_state:
            st.session_state[state_key] = choices[0] if choices else ""
        value = st.selectbox(label, choices, key=state_key, help=help_text)
        if required and (value is None or value == ""):
            st.error(f"{label} is required.")
        return value
    elif param_type.startswith("int"):
        minv = min_value if min_value is not None else 0
        maxv = max_value if max_value is not None else None
        stepv = step if step is not None else 1
        default = default_val if default_val is not None else minv
        if state_key not in st.session_state:
            st.session_state[state_key] = default
        value = st.number_input(label, value=st.session_state[state_key], min_value=minv, max_value=maxv, step=stepv, format="%d", key=state_key, help=help_text)
        if required and value is None:
            st.error(f"{label} is required.")
        return value
    elif param_type.startswith("float"):
        minv = min_value if min_value is not None else 0.0
        maxv = max_value if max_value is not None else None
        stepv = step if step is not None else 0.01
        default = default_val if default_val is not None else minv
        if state_key not in st.session_state:
            st.session_state[state_key] = default
        value = st.number_input(label, value=st.session_state[state_key], min_value=minv, max_value=maxv, step=stepv, format="%f", key=state_key, help=help_text)
        if required and value is None:
            st.error(f"{label} is required.")
        return value
    elif param_type.startswith("list"):
        if state_key not in st.session_state:
            st.session_state[state_key] = default_val or ""
        raw = st.text_area(label, value=st.session_state[state_key], key=state_key, placeholder=placeholder or "Comma-separated or JSON list", help=help_text)
        val = []
        safe_raw = raw if raw is not None else ""
        try:
            val = json.loads(safe_raw)
            if not isinstance(val, list):
                raise ValueError
        except Exception:
            val = [s.strip() for s in safe_raw.split(",") if s.strip()]
        if required and not val:
            st.error(f"{label} is required.")
        return val
    elif param_type.startswith("dict"):
        if state_key not in st.session_state:
            st.session_state[state_key] = default_val or "{}"
        raw = st.text_area(label + " (JSON)", value=st.session_state[state_key], key=state_key, placeholder=placeholder or "JSON object", help=help_text)
        val = {}
        safe_raw = raw if raw is not None else "{}"
        try:
            val = json.loads(safe_raw)
            if not isinstance(val, dict):
                raise ValueError
        except Exception:
            st.warning(f"Invalid JSON for {label}")
            val = {}
        if required and not val:
            st.error(f"{label} is required.")
        return val
    elif param_type.startswith("str"):
        default = default_val if default_val is not None else ("" if param_type == "str" else None)
        if state_key not in st.session_state:
            st.session_state[state_key] = default
        value = st.text_input(label, value=st.session_state[state_key], key=state_key, placeholder=placeholder or "", help=help_text)
        if required and not value:
            st.error(f"{label} is required.")
        return value
    else:
        default = default_val if default_val is not None else ("" if param_type == "str" else None)
        if state_key not in st.session_state:
            st.session_state[state_key] = default
        value = st.text_input(label, value=st.session_state[state_key], key=state_key, placeholder=placeholder or "", help=help_text)
        if required and not value:
            st.error(f"{label} is required.")
        return value
