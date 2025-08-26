import yaml
from typing import Any, Dict


def load_yaml_config(path: str) -> Dict[str, Any]:
    """
    Load and parse a YAML config file, preserving nested structure.
    Args:
        path: Path to the YAML config file.
    Returns:
        Nested dictionary representing the YAML config.
    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML is invalid.
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("YAML config must be a dictionary at the top level.")
    return config
