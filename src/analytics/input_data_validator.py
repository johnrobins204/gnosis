import pandas as pd
from .analytics_addin_manifest_validator import ManifestValidationError

def validate_input_data(data, manifest):
    """
    Validate input data against manifest's input_data_types requirements.
    Supports DataFrame type and required columns with types.
    Raises ManifestValidationError on failure.
    """
    input_types = manifest.get("input_data_types", [])
    if not input_types:
        return  # No requirements
    # Check for DataFrame requirement
    if "DataFrame" in input_types or any(
        isinstance(t, str) and t.lower() == "dataframe" for t in input_types
    ):
        if not isinstance(data, pd.DataFrame):
            raise ManifestValidationError("Input data must be a pandas DataFrame.")
    # Check for required columns and types
    for t in input_types:
        if isinstance(t, dict) and "columns" in t:
            for col in t["columns"]:
                col_name = col["name"]
                col_type = col.get("type")
                if col_name not in data.columns:
                    raise ManifestValidationError(f"Missing required column: {col_name}")
                if col_type:
                    # Only basic type check: float, int, str
                    dtype = str(data[col_name].dtype)
                    if col_type == "float" and not dtype.startswith("float"):
                        raise ManifestValidationError(f"Column {col_name} must be float, got {dtype}")
                    if col_type == "int" and not dtype.startswith("int"):
                        raise ManifestValidationError(f"Column {col_name} must be int, got {dtype}")
                    if col_type == "str" and not dtype.startswith("object"):
                        raise ManifestValidationError(f"Column {col_name} must be str, got {dtype}")
