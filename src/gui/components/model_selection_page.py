import streamlit as st
from model_discovery import list_local_models, list_remote_models
from gui.components.model_selector import render_model_selector

def render_model_selection():
    st.header("Model Selection")
    with st.spinner("Discovering models..."):
        local_models = list_local_models()
        remote_models = list_remote_models()
    def on_select(model):
        st.session_state["selected_model"] = model
        st.success(f"Selected model: {model.get('name') or model.get('id')}")
    render_model_selector(local_models, remote_models, on_select=on_select)
    if "selected_model" in st.session_state:
        st.info(f"Current selection: {st.session_state['selected_model']}")
