import yaml
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional, Union
from gnosis_io import write_provenance
import inference
import judge
import analyst

# Registry for activity handlers and their parameter schemas
ACTIVITY_REGISTRY: dict = {}
print("DEBUG: ACTIVITY_REGISTRY initialized =", ACTIVITY_REGISTRY)

def register_activity(name: str, param_schema: Optional[dict] = None):
    """
    Decorator to register a step/activity handler and its parameter schema.
    """
    def decorator(func):
        ACTIVITY_REGISTRY[name] = {
            "handler": func,
            "param_schema": param_schema or {}
        }
        return func
    return decorator

# Parameter schemas for built-in activities (for GUI config building)
INFERENCE_SCHEMA = {
    "input_csv": {"type": "str", "required": True, "description": "Path to input CSV with 'prompt' column."},
    "output_csv": {"type": "str", "required": True, "description": "Path to write outputs."},
    "default_model": {"type": "str", "required": False, "description": "Default model identifier."},
    "model_config": {"type": "dict", "required": False, "description": "Model config dict."},
    "api_params": {"type": "dict", "required": False, "description": "API params dict."},
    "temperature": {"type": "float", "required": False},
    "top_p": {"type": "float", "required": False},
    "top_k": {"type": "int", "required": False},
    "seed": {"type": "int", "required": False},
    "stop": {"type": "str|list", "required": False},
    "batch_size": {"type": "int", "required": False},
    "output_format": {"type": "str", "required": False},
    "max_tokens": {"type": "int", "required": False},
    "log_level": {"type": "str", "required": False},
}

JUDGE_SCHEMA = {
    "input_csv": {"type": "str", "required": True, "description": "Path to input CSV."},
    "output_csv": {"type": "str", "required": True, "description": "Path to write judged CSV."},
    "template_dir": {"type": "str", "required": True, "description": "Path to judge templates."},
    "rating_col_pattern": {"type": "str", "required": False, "description": "Pattern for rating columns."},
}

ANALYST_SCHEMA = {
    "input_csv": {"type": "str", "required": True, "description": "Path to input CSV."},
    "output_csv": {"type": "str", "required": True, "description": "Path to output CSV."},
    "group_col": {"type": "str|list", "required": True, "description": "Column(s) to group by."},
    "rating_col": {"type": "str", "required": False, "description": "Name of rating column."},
    "metrics": {"type": "list", "required": False, "description": "List of custom metric names."},
}

@register_activity("inference", param_schema=INFERENCE_SCHEMA)
def inference_handler(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return inference.run_from_config(cfg)

@register_activity("judge", param_schema=JUDGE_SCHEMA)
def judge_handler(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return judge.run_from_config(cfg)

@register_activity("analyst", param_schema=ANALYST_SCHEMA)
def analyst_handler(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return analyst.run_from_config(cfg)

print("DEBUG: After decorators, ACTIVITY_REGISTRY =", ACTIVITY_REGISTRY)

def orchestrate(
    config_path: Union[str, Path],
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """
    Load YAML config and execute listed steps sequentially.
    Supports both top-level steps and study.workflow.steps formats.
    Returns dict: {"success": bool, "artifacts": [...], "errors": [...], "study": {...}}
    """
    print(f"DEBUG: orchestrate called with config_path={config_path}")
    if not isinstance(config_path, Path):
        config_path = Path(config_path)
    if not config_path.exists():
        result = {"success": False, "artifacts": [], "errors": [f"config not found: {config_path}"]}
        print(f"DEBUG: returning early, config not found: {result}")
        return result

    with config_path.open("r", encoding="utf-8") as fh:
        try:
            cfg = yaml.safe_load(fh)
            print(f"DEBUG: cfg loaded = {cfg}")
        except Exception as e:
            result = {"success": False, "artifacts": [], "errors": [f"YAML parse error: {e}"]}
            print(f"DEBUG: returning early, YAML error: {result}")
            return result

    # Extract steps from either format
    steps = cfg.get("steps")
    study_meta = {}
    if steps is None:
        study = cfg.get("study", {})
        study_meta = {k: v for k, v in study.items() if k != "workflow"}
        workflow = study.get("workflow", {})
        steps = workflow.get("steps", [])
    print(f"DEBUG: steps = {steps}")
    if not steps:
        result = {"success": False, "artifacts": [], "errors": ["No steps found in config."], "study": study_meta}
        print(f"DEBUG: returning early, no steps: {result}")
        return result

    artifacts: List[str] = []
    errors: List[str] = []
    total_steps = len(steps)

    for idx, step in enumerate(steps):
        name = step.get("name") or step.get("component") or f"step_{idx+1}"
        comp = step.get("component")
        step_cfg = step.get("config", {})
        print(f"DEBUG: step {idx}: name={name}, comp={comp}, step_cfg={step_cfg}")
        reg_entry = ACTIVITY_REGISTRY.get(comp)
        handler = reg_entry["handler"] if reg_entry else None
        print(f"DEBUG: reg_entry={reg_entry}, handler={handler}")

        # Validate step structure
        if not comp:
            error_msg = "missing component or config"
            errors.append(error_msg)
            print(f"DEBUG: appended error: {error_msg}")
            continue
        if not handler:
            error_msg = "unknown component"
            errors.append(error_msg)
            print(f"DEBUG: appended error: {error_msg}")
            continue
        if not step_cfg:
            error_msg = "missing component or config"
            errors.append(error_msg)
            print(f"DEBUG: appended error: {error_msg}")
            continue

        # Progress callback before each step
        if progress_callback:
            progress_callback(idx, total_steps, f"Starting step {idx+1}/{total_steps}: {name}")


        try:
            result = handler(step_cfg)
            print(f"DEBUG: handler result = {result}")
        except Exception as e:
            error_msg = f"{name}: exception during handler execution: {e}"
            errors.append(error_msg)
            print(f"DEBUG: appended error: {error_msg}")
            continue

        # If handler result is not success, or if error key is present, treat as failure
        if not result.get("success") or result.get("error"):
            error_msg = f"{name}: step failed: {result.get('error')}"
            errors.append(error_msg)
            print(f"DEBUG: appended error: {error_msg}")
            continue

        step_artifacts = result.get("artifacts", [])
        for art in step_artifacts:
            try:
                meta = write_provenance(art, {"step": name, "component": comp, "config": step_cfg, **study_meta})
            except Exception:
                meta = None
            artifacts.append(art)
            if meta:
                artifacts.append(meta)
        # Progress callback after each step (optional)
        if progress_callback:
            progress_callback(idx + 1, total_steps, f"Finished step {idx+1}/{total_steps}: {name}")

    final_result = {"success": len(errors) == 0, "artifacts": artifacts, "errors": errors, "study": study_meta}
    print(f"DEBUG: final result = {final_result}")
    return final_result


# Ensure registry and decorator are accessible as module attributes
ACTIVITY_REGISTRY = ACTIVITY_REGISTRY
register_activity = register_activity
__all__ = ["orchestrate", "ACTIVITY_REGISTRY", "register_activity"]
