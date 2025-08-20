import pytest
import pandas as pd
import os
import tempfile

from src.analytics import run_from_config

def test_analytics_smoke():
    """
    Smoke test for the analytics pipeline.
    This test verifies that run_from_config works end-to-end.
    """
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as input_file, \
         tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as output_file:
        
        # Create a minimal test dataset
        test_data = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "prompt": ["Test prompt"] * 4,
            "completion": ["Test response"] * 4,
            "model": ["Llama:8B", "Llama:8B", "gpt-3.5-turbo", "gpt-3.5-turbo"],
            "judge_rating": [4, 5, 3, 4]
        })
        test_data.to_csv(input_file.name, index=False)
        
        # Create a minimal configuration
        config = {
            "input_csv": input_file.name,
            "output_csv": output_file.name,
            "group_by": ["model"]
        }
        
        try:
            # Run analytics
            result = run_from_config(config)
            
            # Verify basic results
            assert result["success"] is True
            assert config["output_csv"] in result["artifacts"]
            
            # Verify output file was created
            assert os.path.exists(output_file.name)
            
            # Check output data
            output_data = pd.read_csv(output_file.name)
            assert len(output_data) == 2  # Two models
            assert "model" in output_data.columns
            assert "count" in output_data.columns
            assert "avg_judge_rating" in output_data.columns
            
            # Verify counts
            llama_row = output_data[output_data["model"] == "Llama:8B"].iloc[0]
            gpt_row = output_data[output_data["model"] == "gpt-3.5-turbo"].iloc[0]
            assert llama_row["count"] == 2
            assert gpt_row["count"] == 2
            
            # Verify averages
            assert llama_row["avg_judge_rating"] == 4.5  # (4+5)/2
            assert gpt_row["avg_judge_rating"] == 3.5  # (3+4)/2
        
        finally:
            # Clean up temporary files
            os.unlink(input_file.name)
            if os.path.exists(output_file.name):
                os.unlink(output_file.name)