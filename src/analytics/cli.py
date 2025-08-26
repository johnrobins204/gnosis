import sys
from typing import Optional
from src.analytics import run_from_config
from src.analytics.yaml_loader import load_yaml_config
from src.analytics.yaml_schema import validate_study_config, ConfigValidationError

def run(argv: Optional[list] = None):
    """
    Run analytics from a YAML config file.
    Usage: python -m src.analytics.cli --config path/to/config.yaml
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run analytics from YAML config")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args(argv)

    from src.logging_config import get_logger
    _logger = get_logger("cli")
    try:
        config = load_yaml_config(args.config)
        from src.analytics.config import apply_env_overrides
        config = apply_env_overrides(config)
        validate_study_config(config)
    except (FileNotFoundError, ValueError, ConfigValidationError) as e:
        _logger.error(f"Config error: {e}")
        sys.exit(1)

    # Pass the loaded config to the analytics pipeline (adapt as needed)
    result = run_from_config(config)
    if not result.get("success", False):
        _logger.error(f"ERROR: {result.get('error')}")
        sys.exit(1)

    _logger.info(f"Analytics complete. Artifacts: {result.get('artifacts')}")
    return result

if __name__ == "__main__":
    run()