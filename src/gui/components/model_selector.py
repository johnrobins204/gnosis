import streamlit as st
from typing import List, Dict, Any


def get_model_options(local_models, remote_models):
    """
    Returns (options, model_map) for use in selectbox and atomic config update.
    """
    options = ["No Override"]
    model_map = {"No Override": None}
    for m in local_models:
        if m.get("error"):
            continue
        provider = m.get("provider", "ollama")
        if provider == "ollama":
            label = f"Local Ollama - {m.get('name') or m.get('id') or str(m)}"
        else:
            label = f"Local {provider.title()} - {m.get('name') or m.get('id') or str(m)}"
        options.append(label)
        m["provider"] = provider
        model_map[label] = m
    for m in remote_models:
        if m.get("error"):
            continue
        provider = m.get("provider", "remote")
        label = f"Remote {provider} - {m.get('name') or m.get('id') or str(m)}"
        options.append(label)
        m["provider"] = provider
        model_map[label] = m
    return options, model_map
