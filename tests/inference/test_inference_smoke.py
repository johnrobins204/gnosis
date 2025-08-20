import pytest
import pandas as pd
import os
import tempfile
from unittest.mock import patch, MagicMock

from src.inference import run_from_config
from src.types import ModelResponse

def test_inference_smoke():
    """
    Smoke test for the inference pipeline with mocked model.
    This test verifies that run_from_config works end-to-end.
    """
    # Create temporary files for input/output
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as input_file, \
         tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as output_file:
        
        # Create a minimal test dataset
        test_data = pd.DataFrame({
            "id": [1],
            "prompt": ["What is 2+2?"]
        })
        test_data.to_csv(input_file.name, index=False)
        
        # Create a minimal configuration
        config = {
            "input_csv": input_file.name,
            "output_csv": output_file.name,
            "default_model": "Llama:8B",
            "model_config": {"provider": "ollama"},
            "api_params": {"server": "http://localhost:11434"}
        }
        
        try:
            # Create a specific mock response
            mock_response = ModelResponse(
                model="Llama:8B",
                prompt="What is 2+2?",
                completion="The answer is 4.",
                metadata={"tokens": 5}
            )
            
            # Create a mock model class
            mock_model = MagicMock()
            mock_model.generate.return_value = mock_response
            
            # Patch get_model_instance directly in the inference module
            with patch("src.inference.get_model_instance") as mock_get_model:
                # Return our mock model
                mock_get_model.return_value = mock_model
                
                # Run inference
                result = run_from_config(config)
                
                # Verify basic results
                assert result["success"] is True
                assert config["output_csv"] in result["artifacts"]
                
                # Verify model was called with the right parameters
                mock_get_model.assert_called()
                mock_model.generate.assert_called_once()
                
                # Verify output file was created
                assert os.path.exists(output_file.name)
                
                # Check output data
                output_data = pd.read_csv(output_file.name)
                assert len(output_data) == 1
                assert "completion" in output_data.columns
                assert output_data["completion"].iloc[0] == "The answer is 4."
        
        finally:
            # Clean up temporary files
            os.unlink(input_file.name)
            if os.path.exists(output_file.name):
                os.unlink(output_file.name)