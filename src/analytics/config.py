
import os
from typing import Dict, Any
from logging_config import get_logger

_logger = get_logger("config")

def apply_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override selected config values with environment variables if present.
    Only DB_URI, API_KEY, and LOG_LEVEL are supported for override.
    Logs a warning if an override occurs.
    """
    overrides = {
        "DB_URI": "db_uri",
        "API_KEY": "api_key",
        "LOG_LEVEL": "log_level",
    }
    for env_var, cfg_key in overrides.items():
        val = os.getenv(env_var)
        if val is not None:
            _logger.warning(f"Overriding config '{cfg_key}' with environment variable '{env_var}'")
            cfg[cfg_key] = val
    return cfg

def validate_config(cfg: Dict[str, Any]) -> None:
    required_keys = ["input_csv", "output_csv", "group_col"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")
    # Optionally, add type checks or more validation here