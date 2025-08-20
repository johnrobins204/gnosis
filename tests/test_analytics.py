import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import tempfile

from src.analytics import run_from_config, detect_rating_columns

@pytest.fixture
def sample_ratings_data():
    """Sample ratings data for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "prompt": ["What is recursion?", "Explain ML", "Define AI"],
        "completion": ["A function calling itself", "Learning from data", "Machine intelligence"],
        "model": ["Llama:8B", "gpt-3.5-turbo", "Llama:8B"],
        "judge_rating": [4, 5, 3]
    })

def test_detect_rating_columns():
    """Test detection of rating columns."""
    # Create DataFrame with various column names
    df = pd.DataFrame({
        "id": [1, 2],
        "prompt": ["foo", "bar"],
        "judge_rating": [4, 5],
        "content_rating": [3, 4],
        "not_a_rating": [1, 2],
        "Rating": [5, 4]
    })
    
    # Test detection
    columns = detect_rating_columns(df)
    
    # Should find these columns
    assert "judge_rating" in columns
    assert "content_rating" in columns
    assert "Rating" in columns
    
    # Should not find this column
    assert "not_a_rating" not in columns

@patch("src.analytics.load_csv")
@patch("src.analytics.write_dataframe")
def test_run_from_config_basic(mock_write, mock_load, sample_ratings_data):
    """Test basic functionality of run_from_config."""
    # Setup mock
    mock_load.return_value = sample_ratings_data
    
    # Create config
    config = {
        "input_csv": "ratings.csv",
        "output_csv": "analysis.csv",
        "group_by": ["model"]
    }
    
    # Run function
    result = run_from_config(config)
    
    # Verify success
    assert result["success"] is True
    assert config["output_csv"] in result["artifacts"]
    
    # Verify dataframe was written
    mock_write.assert_called_once()
    
    # Check the dataframe structure
    written_df = mock_write.call_args[0][0]
    assert "model" in written_df.columns
    assert "count" in written_df.columns
    assert "avg_judge_rating" in written_df.columns

@patch("src.analytics.load_csv")
@patch("src.analytics.write_dataframe")
def test_run_from_config_custom_metrics(mock_write, mock_load, sample_ratings_data):
    """Test run_from_config with custom metrics."""
    # Setup mock
    mock_load.return_value = sample_ratings_data
    
    # Create config with explicit metrics
    config = {
        "input_csv": "ratings.csv",
        "output_csv": "analysis.csv",
        "group_by": ["model"],
        "metrics": ["avg_judge_rating", "count"]
    }
    
    # Run function
    result = run_from_config(config)
    
    # Verify success
    assert result["success"] is True
    
    # Check the dataframe structure
    written_df = mock_write.call_args[0][0]
    assert "avg_judge_rating" in written_df.columns
    assert "count" in written_df.columns

@patch("src.analytics.load_csv")
@patch("src.analytics.write_dataframe")
def test_run_from_config_missing_keys(mock_write, mock_load):
    """Test handling of missing config keys."""
    # Test missing input_csv
    result = run_from_config({"output_csv": "output.csv", "group_by": ["model"]})
    assert result["success"] is False
    assert "missing config key: input_csv" in result["error"]
    
    # Test missing output_csv
    result = run_from_config({"input_csv": "input.csv", "group_by": ["model"]})
    assert result["success"] is False
    assert "missing config key: output_csv" in result["error"]
    
    # Test missing group_by
    result = run_from_config({"input_csv": "input.csv", "output_csv": "output.csv"})
    assert result["success"] is False
    assert "missing config key: group_by" in result["error"]

@patch("src.analytics.load_csv")
def test_run_from_config_empty_data(mock_load):
    """Test handling of empty input data."""
    # Setup mock to return empty DataFrame
    mock_load.return_value = pd.DataFrame()
    
    result = run_from_config({
        "input_csv": "input.csv",
        "output_csv": "output.csv",
        "group_by": ["model"]
    })
    
    assert result["success"] is False
    assert "input data is empty" in result["error"]

@patch("src.analytics.load_csv")
def test_run_from_config_load_error(mock_load):
    """Test handling of errors when loading input file."""
    # Setup mock to raise exception
    mock_load.side_effect = Exception("File not found")
    
    result = run_from_config({
        "input_csv": "nonexistent.csv",
        "output_csv": "output.csv",
        "group_by": ["model"]
    })
    
    assert result["success"] is False
    assert "failed to load input_csv" in result["error"]