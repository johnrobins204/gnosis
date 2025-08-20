import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import json

from src.inference import run_from_config, _row_to_model_id
from src.types import ModelResponse

@pytest.fixture
def sample_config():
    """Sample inference configuration for Ollama server with Llama:8B model."""
    return {
        "input_csv": "test_data.csv",
        "output_csv": "test_output.csv",
        "default_model": "Llama:8B",  # Using the correct key 'default_model'
        "model_config": {"provider": "ollama"},  # Passed to get_model_instance
        "api_params": {"server": "http://localhost:11434"}  # Passed to get_model_instance
    }

@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "prompt": [
            "Explain recursion briefly.",
            "What is machine learning?",
            "Define artificial intelligence."
        ],
        "model": ["Llama:8B", None, "Llama:8B"]
    })

def test_row_to_model_id():
    """Test the _row_to_model_id helper function."""
    # Test with model_id present
    row1 = {"model_id": "Llama:8B", "model": "mistral"}
    assert _row_to_model_id(row1, "default-model") == "Llama:8B"
    
    # Test with only model present
    row2 = {"model": "Llama:8B"}
    assert _row_to_model_id(row2, "default-model") == "Llama:8B"
    
    # Test with neither present
    row3 = {"other_field": "value"}
    assert _row_to_model_id(row3, "default-model") == "default-model"

@patch("src.inference.load_csv")
@patch("src.inference.write_dataframe")
@patch("src.inference.get_model_instance")
def test_run_from_config_basic(mock_get_model, mock_write, mock_load, sample_config, sample_data):
    """Test basic functionality of run_from_config with Ollama/Llama:8B."""
    # Setup mocks
    mock_load.return_value = sample_data
    mock_model = MagicMock()
    
    # Configure each call to generate to return a single ModelResponse
    mock_model.generate.side_effect = [
        ModelResponse(
            model="Llama:8B",
            prompt="Explain recursion briefly.",
            completion="Recursion is a concept where a function calls itself.",
            metadata={"tokens": 18}
        ),
        ModelResponse(
            model="Llama:8B",
            prompt="What is machine learning?",
            completion="Machine learning is a field of AI that enables systems to learn.",
            metadata={"tokens": 20}
        ),
        ModelResponse(
            model="Llama:8B",
            prompt="Define artificial intelligence.",
            completion="Artificial intelligence is the simulation of human intelligence.",
            metadata={"tokens": 17}
        )
    ]
    mock_get_model.return_value = mock_model
    
    # Run the function
    result = run_from_config(sample_config)
    
    # Verify calls with correct parameters
    mock_load.assert_called_once_with(sample_config["input_csv"])
    assert mock_get_model.call_count == 3  # Called once per row
    mock_write.assert_called_once()
    
    # Verify result structure
    assert result["success"] is True
    assert sample_config["output_csv"] in result["artifacts"]
    
    # Verify number of rows processed
    assert mock_model.generate.call_count == 3  # Called once per row

@patch("src.inference.load_csv")
@patch("src.inference.write_dataframe")
@patch("src.inference.get_model_instance")
def test_run_from_config_error_handling(mock_get_model, mock_write, mock_load, sample_config, sample_data):
    """Test error handling in run_from_config."""
    # Setup mocks
    mock_load.return_value = sample_data
    mock_model = MagicMock()
    
    # First and third calls succeed, second call raises an exception
    mock_model.generate.side_effect = [
        ModelResponse(
            model="Llama:8B",
            prompt="Explain recursion briefly.",
            completion="Recursion is a concept where a function calls itself.",
            metadata={"tokens": 18}
        ),
        Exception("Ollama API error: context length exceeded"),
        ModelResponse(
            model="Llama:8B",
            prompt="Define artificial intelligence.",
            completion="Artificial intelligence is the simulation of human intelligence.",
            metadata={"tokens": 17}
        )
    ]
    mock_get_model.return_value = mock_model
    
    # Run the function
    result = run_from_config(sample_config)
    
    # Verify success - function handles errors and continues
    assert result["success"] is True
    assert sample_config["output_csv"] in result["artifacts"]
    
    # Verify all rows were processed
    assert mock_model.generate.call_count == 3
    
    # Check the dataframe that was written to verify error handling
    write_df_arg = mock_write.call_args[0][0]  # First arg to write_dataframe
    
    # The second row should have empty completion and error metadata
    assert len(write_df_arg) == 3
    assert write_df_arg.iloc[1]["completion"] == ""
    assert "error" in json.loads(write_df_arg.iloc[1]["metadata"])

@patch("src.inference.load_csv")
@patch("src.inference.write_dataframe")
def test_run_from_config_missing_keys(mock_write, mock_load):
    """Test handling of missing config keys."""
    # Test missing input_csv
    result = run_from_config({"output_csv": "output.csv"})
    assert result["success"] is False
    assert "missing config key: input_csv" in result["error"]
    
    # Test missing output_csv
    result = run_from_config({"input_csv": "input.csv"})
    assert result["success"] is False
    assert "missing config key: output_csv" in result["error"]

@patch("src.inference.load_csv")
def test_run_from_config_missing_prompt_column(mock_load):
    """Test handling of missing prompt column in input data."""
    # Setup mock to return DataFrame without prompt column
    mock_load.return_value = pd.DataFrame({"id": [1, 2], "text": ["foo", "bar"]})
    
    result = run_from_config({
        "input_csv": "input.csv",
        "output_csv": "output.csv"
    })
    
    assert result["success"] is False
    assert "missing required 'prompt' column" in result["error"]

@patch("src.inference.load_csv")
def test_run_from_config_load_error(mock_load):
    """Test handling of errors when loading input file."""
    # Setup mock to raise exception
    mock_load.side_effect = Exception("File not found")
    
    result = run_from_config({
        "input_csv": "nonexistent.csv",
        "output_csv": "output.csv"
    })
    
    assert result["success"] is False
    assert "failed to load input_csv" in result["error"]