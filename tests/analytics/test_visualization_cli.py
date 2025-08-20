import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd

from src.analytics.visualization import run

def test_cli_basic_usage():
    """Test basic CLI usage with required arguments."""
    test_args = [
        "visualization.py",
        "--input", "test_input.csv",
        "--output-dir", "test_output",
        "--type", "comparative",
        "--metric", "score"
    ]
    
    with patch("sys.argv", test_args), \
         patch("src.analytics.visualization.run_from_config") as mock_run, \
         patch("sys.exit") as mock_exit:
        
        # Set up successful return value
        mock_run.return_value = {"success": True, "artifacts": ["test_output/comparative_score.png"]}
        
        # Run the CLI
        run()
        
        # Verify correct config was passed
        mock_run.assert_called_once()
        config = mock_run.call_args[0][0]
        assert config["input_csv"] == "test_input.csv"
        assert config["output_dir"] == "test_output"
        assert config["visualization_type"] == "comparative"
        assert config["metric_name"] == "score"
        
        # Verify sys.exit was not called
        mock_exit.assert_not_called()

def test_cli_with_optional_args():
    """Test CLI with optional arguments."""
    test_args = [
        "visualization.py",
        "--input", "test_input.csv",
        "--output-dir", "test_output",
        "--type", "statistical",
        "--metric", "score",
        "--group-by", "category",
        "--hue", "subcategory",
        "--theme", "custom"
    ]
    
    with patch("sys.argv", test_args), \
         patch("src.analytics.visualization.run_from_config") as mock_run, \
         patch("sys.exit") as mock_exit:
        
        # Set up successful return value
        mock_run.return_value = {"success": True, "artifacts": ["test_output/statistical_score.png"]}
        
        # Run the CLI
        run()
        
        # Verify optional parameters were passed correctly
        config = mock_run.call_args[0][0]
        assert config["group_by"] == "category"
        assert config["hue"] == "subcategory"
        assert config["theme"] == "custom"
        
        # Verify sys.exit was not called
        mock_exit.assert_not_called()

def test_cli_error_handling():
    """Test CLI error handling."""
    test_args = [
        "visualization.py",
        "--input", "nonexistent.csv",
        "--output-dir", "test_output",
        "--type", "comparative",
        "--metric", "score"
    ]
    
    with patch("sys.argv", test_args), \
         patch("src.analytics.visualization.run_from_config") as mock_run, \
         patch("sys.exit") as mock_exit:
        
        # Set up error return value
        mock_run.return_value = {"success": False, "error": "File not found"}
        
        # Run the CLI
        run()
        
        # Verify exit was called with error code
        mock_exit.assert_called_once_with(1)