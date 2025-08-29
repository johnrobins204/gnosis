import os
from datetime import datetime

def get_default_output_path(tab_state=None):
    env_path = os.environ.get("GNOSIS_OUTPUT_PATH")
    if env_path:
        return env_path
    workflow_name = None
    if tab_state and "study" in tab_state and "name" in tab_state["study"]:
        workflow_name = tab_state["study"]["name"].replace(" ", "_")
    else:
        workflow_name = "workplan"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suggested = f"outputs/{workflow_name}_{timestamp}.csv"
    return suggested
