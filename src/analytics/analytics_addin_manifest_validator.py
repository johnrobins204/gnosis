# analytics_addin_manifest_validator.py
"""
Validator for analytics add-in manifest files (YAML/JSON).
Checks required fields, types, and values per the manifest schema.
"""
import yaml

REQUIRED_FIELDS = [
    "name",
    "version",
    "author",
    "description",
    "input_data_types",
    "metrics",
    "dependencies",
    "entry_point",
]

class ManifestValidationError(Exception):
    pass

def validate_manifest(manifest_path):
    """
    Validate an analytics add-in manifest file (YAML).
    Raises ManifestValidationError on failure.
    Returns parsed manifest dict on success.
    """
    with open(manifest_path, 'r') as f:
        manifest = yaml.safe_load(f)
    missing = [field for field in REQUIRED_FIELDS if field not in manifest]
    if missing:
        raise ManifestValidationError(f"Missing required fields: {missing}")
    # Type checks (basic)
    if not isinstance(manifest["name"], str):
        raise ManifestValidationError("'name' must be a string")
    if not isinstance(manifest["version"], str):
        raise ManifestValidationError("'version' must be a string")
    if not isinstance(manifest["author"], str):
        raise ManifestValidationError("'author' must be a string")
    if not isinstance(manifest["description"], str):
        raise ManifestValidationError("'description' must be a string")
    if not isinstance(manifest["input_data_types"], list):
        raise ManifestValidationError("'input_data_types' must be a list")
    if not isinstance(manifest["metrics"], list):
        raise ManifestValidationError("'metrics' must be a list")
    if not isinstance(manifest["dependencies"], list):
        raise ManifestValidationError("'dependencies' must be a list")
    if not isinstance(manifest["entry_point"], str):
        raise ManifestValidationError("'entry_point' must be a string")
    # Optionally: check metrics structure
    for metric in manifest["metrics"]:
        if not isinstance(metric, dict) or "name" not in metric:
            raise ManifestValidationError("Each metric must be a dict with a 'name' field")
    return manifest
