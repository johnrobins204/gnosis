
import streamlit as st
from gui.components.fields import render_config_field
from orchestrator import ACTIVITY_REGISTRY
import yaml
import json
import os

from gui.utils import get_default_output_path
from gui.state import WorkflowState, dispatch_action

def render_data_source_form(tab_state, steps):
    st.subheader("Data Source Configuration")
    # Only one data source config allowed, always at index 0
    # Use canonical WorkflowState from session
    ws = st.session_state.get("workflow_state")
    if ws is None:
        ws = WorkflowState()
        st.session_state["workflow_state"] = ws
    # Ensure data_source block exists
    ds_cfg = None
    if ws.workflow_steps and ws.workflow_steps[0].get("component") == "data_source":
        ds_cfg = ws.workflow_steps[0]["config"]
    else:
        ds_cfg = {
            "dataset_path": "data/input.csv",
            "description": "A CSV file containing input data for the workflow. Example: survey_results.csv with columns: id, question, answer."
        }
        ws.workflow_steps.insert(0, {"component": "data_source", "config": ds_cfg})
        st.session_state["workflow_state"] = ws

    def update_ds_cfg_dataset_path():
        dispatch_action({
            "type": "UPDATE_FIELD",
            "payload": {
                "block": "data_source",
                "field": "dataset_path",
                "value": st.session_state["dataset_path_field"]
            }
        })
    def update_ds_cfg_description():
        dispatch_action({
            "type": "UPDATE_FIELD",
            "payload": {
                "block": "data_source",
                "field": "description",
                "value": st.session_state["dataset_desc_field"]
            }
        })
    st.text_input(
        "Dataset File Path or URI",
        value=ds_cfg.get("dataset_path", ""),
        key="dataset_path_field",
        on_change=update_ds_cfg_dataset_path
    )
    st.text_area(
        "Dataset Description or Metadata",
        value=ds_cfg.get("description", ""),
        height=60,
        key="dataset_desc_field",
        on_change=update_ds_cfg_description
    )

def render_inference_block_form(tab_state, steps):
    st.subheader("Inference Block Configuration")
    ws_dict = st.session_state.get("workflow_state")
    if ws_dict is None:
        ws = WorkflowState()
        st.session_state["workflow_state"] = ws.to_dict()
    else:
        ws = WorkflowState.from_dict(ws_dict)
    # Find or create inference_block
    inf_block = None
    for s in ws.workflow_steps:
        if s.get("component") == "inference_block":
            inf_block = s
            break
    if inf_block is None:
        inf_block = {"component": "inference_block", "configs": []}
        ws.workflow_steps.append(inf_block)
        st.session_state["workflow_state"] = ws.to_dict()
    inference_cfgs = inf_block["configs"]
    from model_discovery import list_local_models, list_remote_models
    from gui.components import model_selector
    local_models = list_local_models()
    remote_models = list_remote_models()
    default_inf_cfg = {
        "name": "Default Inference",
        "description": "Run LLM inference on each row.",
        "model_override": None,
        "model": None,
        "model_config_override": '{"temperature": 0.2}',
        "prompt_text": "What is the main topic of this answer?",
        "prompt_json": '{"instruction": "Classify the answer."}'
    }
    for i, inf_cfg in enumerate(inference_cfgs):
        for k, v in default_inf_cfg.items():
            if inf_cfg.get(k) is None:
                inf_cfg[k] = v
        with st.expander(f"Inference Config {i+1}", expanded=False):
            def update_inf_cfg_field(field):
                def _cb(idx=i, field=field):
                    dispatch_action({
                        "type": "UPDATE_FIELD",
                        "payload": {
                            "block": "inference_block",
                            "index": idx,
                            "field": field,
                            "value": st.session_state[f"inference_block_{field}_{idx}"]
                        }
                    })
                return _cb
            def update_inf_cfg_model_override(idx=i):
                selected_label = st.session_state.get(f"inference_block_model_override_{idx}")
                model_options, model_map = model_selector.get_model_options(local_models, remote_models)
                selected_model = model_map.get(selected_label) if selected_label in model_map else None
                dispatch_action({
                    "type": "UPDATE_FIELD",
                    "payload": {
                        "block": "inference_block",
                        "index": idx,
                        "field": "model_override",
                        "value": selected_model.get("api_path") or selected_model.get("id") if selected_model else None
                    }
                })
                dispatch_action({
                    "type": "UPDATE_FIELD",
                    "payload": {
                        "block": "inference_block",
                        "index": idx,
                        "field": "model",
                        "value": selected_model.get("id") if selected_model else None
                    }
                })
            st.text_input(
                f"Config Name {i+1}",
                value=inf_cfg.get("name", ""),
                key=f"inference_block_name_{i}",
                on_change=update_inf_cfg_field("name")
            )
            st.text_area(
                f"Description {i+1}",
                value=inf_cfg.get("description", ""),
                height=40,
                key=f"inference_block_desc_{i}",
                on_change=update_inf_cfg_field("description")
            )
            model_options, model_map = model_selector.get_model_options(local_models, remote_models)
            default_label = model_options[0]
            current_label = default_label
            if inf_cfg.get("model_override") is not None:
                for label, m in model_map.items():
                    if m and (m.get("api_path") == inf_cfg["model_override"] or m.get("id") == inf_cfg["model_override"]):
                        current_label = label
                        break
            st.selectbox(
                "Model Override",
                model_options,
                index=model_options.index(current_label) if current_label in model_options else 0,
                key=f"inference_block_model_override_{i}",
                on_change=update_inf_cfg_model_override
            )
            st.text_area(
                f"Override Model Config (JSON, optional) {i+1}",
                value=inf_cfg.get("model_config_override", ""),
                height=60,
                key=f"inference_block_model_config_override_{i}",
                on_change=update_inf_cfg_field("model_config_override")
            )
            st.text_area(
                f"Prompt Text {i+1}",
                value=inf_cfg.get("prompt_text", ""),
                height=60,
                key=f"inference_block_prompt_text_{i}",
                on_change=update_inf_cfg_field("prompt_text")
            )
            st.text_area(
                f"Prompt JSON {i+1}",
                value=inf_cfg.get("prompt_json", ""),
                height=60,
                key=f"inference_block_prompt_json_{i}",
                on_change=update_inf_cfg_field("prompt_json")
            )
            # Remove button (now reducerized)
            if st.button(f"Remove", key=f"remove_inference_cfg_{i}"):
                dispatch_action({
                    "type": "REMOVE_BLOCK",
                    "payload": {
                        "block": "inference_block",
                        "index": i
                    }
                })
    # Add new config block (now reducerized)
    if st.button("Add Inference Config Block", key="add_inference_cfg_block"):
        dispatch_action({
            "type": "ADD_BLOCK",
            "payload": {
                "block": "inference_block"
            }
        })

def render_judge_block_form(tab_state, steps):
    st.subheader("Judge Block Configuration")
    ws_dict = st.session_state.get("workflow_state")
    if ws_dict is None:
        ws = WorkflowState()
        st.session_state["workflow_state"] = ws.to_dict()
    else:
        ws = WorkflowState.from_dict(ws_dict)
    # Find or create judge_block
    judge_block = None
    for s in ws.workflow_steps:
        if s.get("component") == "judge_block":
            judge_block = s
            break
    if judge_block is None:
        judge_block = {"component": "judge_block", "configs": []}
        ws.workflow_steps.append(judge_block)
        st.session_state["workflow_state"] = ws.to_dict()
    judge_cfgs = judge_block["configs"]
    template_dir = os.path.join('configs', 'injection_templates')
    if not os.path.isdir(template_dir):
        st.error(f"Judge template directory not found: {template_dir}. Please ensure the directory exists and contains judge_*.json files.")
        return
    template_files = [f for f in os.listdir(template_dir) if f.startswith('judge_') and f.endswith('.json')]
    template_files.sort()
    template_labels = []
    template_jsons = []
    for f in template_files:
        with open(os.path.join(template_dir, f), 'r') as tf:
            j = json.load(tf)
            if isinstance(j.get('evaluationCriteria'), dict) and len(j['evaluationCriteria']) > 0:
                label = list(j['evaluationCriteria'].keys())[0]
            else:
                label = f.replace('judge_', '').replace('.json', '').replace('_', ' ').title()
            template_labels.append(label)
            template_jsons.append(j)
    default_judge_cfg = {
        "template_file": None,
        "template_label": None,
        "default_model": "gpt-3.5-turbo"
    }
    for i, judge_cfg in enumerate(judge_cfgs):
        for k, v in default_judge_cfg.items():
            if judge_cfg.get(k) is None:
                judge_cfg[k] = v
        with st.expander(f"Judge Config {i+1}", expanded=False):
            def update_judge_cfg_field(field):
                def _cb(idx=i, field=field):
                    dispatch_action({
                        "type": "UPDATE_FIELD",
                        "payload": {
                            "block": "judge_block",
                            "index": idx,
                            "field": field,
                            "value": st.session_state.get(f"judge_block_{field}_{idx}")
                        }
                    })
                return _cb
            def update_judge_cfg_template(idx=i):
                selected_idx = st.session_state.get(f"judge_block_template_select_{idx}")
                if selected_idx and selected_idx in template_labels:
                    selected_file = template_files[template_labels.index(selected_idx)]
                else:
                    selected_file = None
                dispatch_action({
                    "type": "UPDATE_FIELD",
                    "payload": {
                        "block": "judge_block",
                        "index": idx,
                        "field": "template_file",
                        "value": selected_file
                    }
                })
                dispatch_action({
                    "type": "UPDATE_FIELD",
                    "payload": {
                        "block": "judge_block",
                        "index": idx,
                        "field": "template_label",
                        "value": selected_idx
                    }
                })
            if template_labels:
                selected_idx = st.selectbox(
                    f"Judge Template {i+1}",
                    template_labels,
                    index=template_labels.index(judge_cfg.get("template_label", template_labels[0])) if judge_cfg.get("template_label") in template_labels else 0,
                    key=f"judge_block_template_select_{i}",
                    on_change=update_judge_cfg_template
                )
                selected_file = template_files[template_labels.index(selected_idx)]
                with open(os.path.join(template_dir, selected_file), 'r') as f:
                    selected_template_json = f.read()
                st.text_area(f"Template Preview (JSON) {i+1}", value=selected_template_json, height=150, key=f"judge_block_template_preview_{i}")
            else:
                pass
            st.text_input(
                f"Judge Model (ID or Name) {i+1}",
                value=judge_cfg.get("default_model", ""),
                key=f"judge_block_model_field_{i}",
                on_change=update_judge_cfg_field("default_model")
            )
            if st.button(f"Remove", key=f"remove_judge_cfg_{i}"):
                dispatch_action({
                    "type": "REMOVE_BLOCK",
                    "payload": {
                        "block": "judge_block",
                        "index": i
                    }
                })
    if st.button("Add Judge Config Block", key="add_judge_cfg_block"):
        dispatch_action({
            "type": "ADD_BLOCK",
            "payload": {
                "block": "judge_block"
            }
        })

def render_analytics_block_form(tab_state, steps):
    st.subheader("Analytics Block Configuration")
    ws_dict = st.session_state.get("workflow_state")
    if ws_dict is None:
        ws = WorkflowState()
        st.session_state["workflow_state"] = ws.to_dict()
    else:
        ws = WorkflowState.from_dict(ws_dict)
    # Find or create analytics_block
    analytics_block = None
    for s in ws.workflow_steps:
        if s.get("component") == "analytics_block":
            analytics_block = s
            break
    if analytics_block is None:
        analytics_block = {"component": "analytics_block", "configs": []}
        ws.workflow_steps.append(analytics_block)
        st.session_state["workflow_state"] = ws.to_dict()
    analytics_cfgs = analytics_block["configs"]
    default_analytics_cfg = {
        "name": "Basic Analytics",
        "description": "Aggregate and summarize results."
    }
    for i, analytics_cfg in enumerate(analytics_cfgs):
        for k, v in default_analytics_cfg.items():
            if analytics_cfg.get(k) is None:
                analytics_cfg[k] = v
        with st.expander(f"Analytics Config {i+1}", expanded=False):
            def update_analytics_cfg_field(field):
                def _cb(idx=i, field=field):
                    dispatch_action({
                        "type": "UPDATE_FIELD",
                        "payload": {
                            "block": "analytics_block",
                            "index": idx,
                            "field": field,
                            "value": st.session_state.get(f"analytics_block_{field}_field_{idx}")
                        }
                    })
                return _cb
            st.text_input(
                f"Analytics Name {i+1}",
                value=analytics_cfg.get("name", ""),
                key=f"analytics_block_name_field_{i}",
                on_change=update_analytics_cfg_field("name")
            )
            st.text_area(
                f"Description {i+1}",
                value=analytics_cfg.get("description", ""),
                key=f"analytics_block_desc_field_{i}",
                on_change=update_analytics_cfg_field("description")
            )
            if st.button(f"Remove", key=f"remove_analytics_cfg_{i}"):
                dispatch_action({
                    "type": "REMOVE_BLOCK",
                    "payload": {
                        "block": "analytics_block",
                        "index": i
                    }
                })
    if st.button("Add Analytics Config Block", key="add_analytics_cfg_block"):
        dispatch_action({
            "type": "ADD_BLOCK",
            "payload": {
                "block": "analytics_block"
            }
        })