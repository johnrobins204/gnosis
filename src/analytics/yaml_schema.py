from typing import Any, Dict, List

class ConfigValidationError(Exception):
    pass

def validate_study_config(cfg: Dict[str, Any]) -> None:
    """
    Validate the loaded YAML config against the expected analytics schema.
    Raises ConfigValidationError with a clear message if invalid.
    """
    if not isinstance(cfg, dict):
        raise ConfigValidationError("Config must be a dictionary at the top level.")
    if "study" not in cfg:
        raise ConfigValidationError("Missing required top-level key: 'study'")
    study = cfg["study"]
    if not isinstance(study, dict):
        raise ConfigValidationError("'study' must be a dictionary.")
    if "workflow" not in study:
        raise ConfigValidationError("Missing required key: 'workflow' in 'study'")
    workflow = study["workflow"]
    if not isinstance(workflow, dict):
        raise ConfigValidationError("'workflow' must be a dictionary.")
    if "steps" not in workflow:
        raise ConfigValidationError("Missing required key: 'steps' in 'workflow'")
    steps = workflow["steps"]
    if not isinstance(steps, list) or not steps:
        raise ConfigValidationError("'steps' in 'workflow' must be a non-empty list.")
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ConfigValidationError(f"Step {i} in 'workflow.steps' must be a dictionary.")
        if "component" not in step:
            raise ConfigValidationError(f"Step {i} missing required key: 'component'.")
        if "config" not in step:
            raise ConfigValidationError(f"Step {i} missing required key: 'config'.")
        if not isinstance(step["config"], dict):
            raise ConfigValidationError(f"'config' in step {i} must be a dictionary.")
    # Optionally, add more detailed validation for each component/config
