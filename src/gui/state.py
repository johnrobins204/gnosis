import streamlit as st
from typing import Any, Dict

# Canonical workflow state object

import streamlit as st
from typing import Any, Dict

# Canonical workflow state object
class WorkflowState:
    def __init__(self, study=None, workflow_steps=None, workplan_active=True):
        self.study = study if study is not None else {"name": "My Study", "description": "", "workflow": {"steps": []}}
        self.workflow_steps = workflow_steps if workflow_steps is not None else []
        self.workplan_active = workplan_active

    def to_dict(self):
        return {
            "study": self.study,
            "workflow_steps": self.workflow_steps,
            "workplan_active": self.workplan_active
        }

    @staticmethod
    def from_dict(d):
        return WorkflowState(
            study=d.get("study"),
            workflow_steps=d.get("workflow_steps"),
            workplan_active=d.get("workplan_active", True)
        )

# Action types: UPDATE_FIELD, ADD_BLOCK, REMOVE_BLOCK, etc.
def workflow_reducer(state: WorkflowState, action: Dict[str, Any]) -> WorkflowState:
    t = action.get("type")
    payload = action.get("payload", {})
    # Shallow copy for immutability
    import copy
    new_state = WorkflowState(
        study=copy.deepcopy(state.study),
        workflow_steps=copy.deepcopy(state.workflow_steps),
        workplan_active=state.workplan_active
    )
    if t == "UPDATE_FIELD":
        block = payload.get("block")
        index = payload.get("index", 0)
        field = payload.get("field")
        value = payload.get("value")
        if block == "data_source":
            if new_state.workflow_steps and new_state.workflow_steps[0].get("component") == "data_source":
                new_state.workflow_steps[0]["config"][field] = value
        else:
            for step in new_state.workflow_steps:
                if step.get("component") == block:
                    if "configs" in step and 0 <= index < len(step["configs"]):
                        step["configs"][index][field] = value
                    break
    if t == "ADD_BLOCK":
        block = payload.get("block")
        if block in ("inference_block", "judge_block", "analytics_block"):
            default_cfg = {"inference_block": {"name": "", "description": ""},
                          "judge_block": {"template_file": None, "default_model": ""},
                          "analytics_block": {"name": "", "description": ""}}[block]
            for step in new_state.workflow_steps:
                if step.get("component") == block:
                    step.setdefault("configs", []).append(default_cfg.copy())
                    break
    if t == "REMOVE_BLOCK":
        block = payload.get("block")
        index = payload.get("index", 0)
        if block in ("inference_block", "judge_block", "analytics_block"):
            for step in new_state.workflow_steps:
                if step.get("component") == block:
                    if "configs" in step and 0 <= index < len(step["configs"]):
                        step["configs"].pop(index)
                    break
    return new_state

def dispatch_action(action):
    ws_dict = st.session_state.get("workflow_state")
    if ws_dict is None:
        ws = WorkflowState()
    else:
        ws = WorkflowState.from_dict(ws_dict)
    new_ws = workflow_reducer(ws, action)
    st.session_state["workflow_state"] = new_ws.to_dict()
    try:
        import streamlit as st
        st.rerun()
    except AttributeError:
        pass
def create_new_tab(st, tab_name=None, study_obj=None, WORKPLAN_TAB_CAP=5):
    if len(st.session_state["workplan_tabs"]) >= WORKPLAN_TAB_CAP:
        st.sidebar.warning(f"Tab cap of {WORKPLAN_TAB_CAP} reached. Close a tab before creating a new one.")
        return
    if not tab_name:
        tab_name = f"Workplan {len(st.session_state['workplan_tabs'])+1}"
    if tab_name in st.session_state["workplan_tabs"]:
        return
    st.session_state["workplan_tabs"].append(tab_name)
    st.session_state["active_tab"] = tab_name
    st.session_state["tab_states"][tab_name] = {
        "study": study_obj if study_obj else {"name": "My Study", "description": "", "workflow": {"steps": []}},
        "workflow_steps": study_obj.get("workflow", {}).get("steps", []) if study_obj else [],
        "workplan_active": True
    }

def close_active_tab(st):
    tab_to_close = st.session_state["active_tab"]
    st.session_state["workplan_tabs"].remove(tab_to_close)
    st.session_state["tab_states"].pop(tab_to_close, None)
    if st.session_state["workplan_tabs"]:
        st.session_state["active_tab"] = st.session_state["workplan_tabs"][0]
    else:
        st.session_state["active_tab"] = None
    st.session_state["show_tab_closed"] = True

def get_tab_state(st):
    tab = st.session_state.get("active_tab")
    if not tab:
        return None
    return st.session_state["tab_states"].get(tab)

def sync_workplan_active(st):
    tabs = st.session_state.get("workplan_tabs", [])
    active_tab = st.session_state.get("active_tab")
    tab_states = st.session_state.get("tab_states", {})
    if tabs and active_tab in tab_states:
        tab_states[active_tab]["workplan_active"] = True
    else:
        st.session_state["workplan_active"] = False
