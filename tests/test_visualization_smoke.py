import pytest
import pandas as pd
import os
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from src.analytics.visualization import run_from_config

def test_visualization_smoke():
    """
    Smoke test for the visualization module.
    This test verifies that run_from_config works end-to-end.
    """
    # Create temporary files and directories
    with tempfile.TemporaryDirectory() as temp_dir, \
         tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as input_file:
        
        # Create output directory path
        output_dir = os.path.join(temp_dir, "output")
        
        # Create a minimal test dataset
        test_data = pd.DataFrame({
            "model": ["Model A", "Model B", "Model A", "Model B"],
            "prompt": ["Prompt 1", "Prompt 1", "Prompt 2", "Prompt 2"],
            "score": [0.8, 0.7, 0.9, 0.6],
            "accuracy": [0.9, 0.8, 0.7, 0.6]
        })
        test_data.to_csv(input_file.name, index=False)
        
        try:
            # Test comparative visualization
            config = {
                "input_csv": input_file.name,
                "output_dir": output_dir,
                "visualization_type": "comparative",
                "metric_name": "score"
            }
            
            # Run visualization
            result = run_from_config(config)
            
            # Verify basic results
            assert result["success"] is True
            assert len(result["artifacts"]) == 1
            
            # Verify output file was created
            assert os.path.exists(result["artifacts"][0])
            
            # Test radar visualization
            config = {
                "input_csv": input_file.name,
                "output_dir": output_dir,
                "visualization_type": "radar"
            }
            
            # Run visualization
            result = run_from_config(config)
            
            # Verify basic results
            assert result["success"] is True
            assert len(result["artifacts"]) == 1
            
            # Verify output file was created
            assert os.path.exists(result["artifacts"][0])
            
        finally:
            # Clean up temporary file
            os.unlink(input_file.name)