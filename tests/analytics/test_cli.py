import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd

from src.analytics.cli import run

def test_cli_basic_usage():
    """Test basic CLI usage with required arguments."""
    test_args = [
        "analytics.py",
        "--input", "test_input.csv",
        "--output", "test_output.csv",
        "--group-by", "model"
    ]
    
    # Need to patch both run_from_config AND load_csv
    with patch("sys.argv", test_args), \
         patch("src.analytics.cli.run_from_config") as mock_run, \
         patch("sys.exit") as mock_exit:
        
        # Set up successful return value
        mock_run.return_value = {"success": True, "artifacts": ["test_output.csv"]}
        
        # Run the CLI
        run()
        
        # Verify correct config was passed
        mock_run.assert_called_once()
        config = mock_run.call_args[0][0]
        assert config["input_csv"] == "test_input.csv"
        assert config["output_csv"] == "test_output.csv"
        assert config["group_by"] == ["model"]
        
        # Verify sys.exit was not called
        mock_exit.assert_not_called()

def test_cli_with_metrics():
    """Test CLI with specified metrics."""
    test_args = [
        "analytics.py",
        "--input", "test_input.csv",
        "--output", "test_output.csv",
        "--group-by", "model",
        "--metrics", "avg_judge_rating,count"
    ]
    
    with patch("sys.argv", test_args), \
         patch("src.analytics.cli.run_from_config") as mock_run, \
         patch("sys.exit") as mock_exit:
        
        # Set up successful return value
        mock_run.return_value = {"success": True, "artifacts": ["test_output.csv"]}
        
        # Run the CLI
        run()
        
        # Verify metrics were parsed correctly
        config = mock_run.call_args[0][0]
        assert "metrics" in config
        assert config["metrics"] == ["avg_judge_rating", "count"]
        
        # Verify sys.exit was not called
        mock_exit.assert_not_called()

def test_cli_error_handling():
    """Test CLI error handling."""
    test_args = [
        "analytics.py",
        "--input", "nonexistent.csv",
        "--output", "test_output.csv",
        "--group-by", "model"
    ]
    
    with patch("sys.argv", test_args), \
         patch("src.analytics.cli.run_from_config") as mock_run, \
         patch("sys.exit") as mock_exit:
        
        # Set up error return value
        mock_run.return_value = {"success": False, "error": "File not found"}
        
        # Run the CLI
        run()
        
        # Verify exit was called with error code
        mock_exit.assert_called_once_with(1)