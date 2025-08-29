import json
import io
import pandas as pd
import streamlit as st

def parse_questions_file(uploaded_file):
    """
    Try to parse the uploaded file as JSON. If it fails, return None and an error message.
    Returns: (data, format, error)
    - data: parsed data (list/dict/DataFrame)
    - format: 'json' if successful, else None
    - error: error message if failed, else None
    """
    try:
        # Try to decode as UTF-8 text
        text = uploaded_file.read().decode("utf-8")
        # Try to parse as JSON
        data = json.loads(text)
        return data, 'json', None
    except Exception as e:
        return None, None, f"Failed to parse as JSON: {e}"
