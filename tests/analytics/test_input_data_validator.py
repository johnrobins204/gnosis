import pytest
import pandas as pd
from src.analytics.input_data_validator import validate_input_data
from src.analytics.analytics_addin_manifest_validator import ManifestValidationError

BASIC_MANIFEST = {
    "input_data_types": [
        "DataFrame",
        {"columns": [
            {"name": "score", "type": "float"},
            {"name": "label", "type": "str"}
        ]}
    ]
}

def test_valid_input():
    df = pd.DataFrame({"score": [1.0, 2.5], "label": ["a", "b"]})
    validate_input_data(df, BASIC_MANIFEST)  # Should not raise

def test_missing_column():
    df = pd.DataFrame({"score": [1.0, 2.5]})
    with pytest.raises(ManifestValidationError) as e:
        validate_input_data(df, BASIC_MANIFEST)
    assert "Missing required column: label" in str(e.value)

def test_wrong_type():
    df = pd.DataFrame({"score": [1, 2], "label": ["a", "b"]})  # score is int, not float
    with pytest.raises(ManifestValidationError) as e:
        validate_input_data(df, BASIC_MANIFEST)
    assert "Column score must be float" in str(e.value)

def test_not_dataframe():
    data = [1, 2, 3]
    with pytest.raises(ManifestValidationError) as e:
        validate_input_data(data, BASIC_MANIFEST)
    assert "Input data must be a pandas DataFrame" in str(e.value)
