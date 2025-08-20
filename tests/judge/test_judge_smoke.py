import pytest
import pandas as pd
import os
import json
import tempfile

from src.judge import run_from_config

def test_judge_smoke():
    """
    Smoke test for the judge pipeline.
    This test verifies that run_from_config works end-to-end.
    """
    # Create temporary files and directories
    with tempfile.TemporaryDirectory() as temp_dir, \
         tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as input_file, \
         tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as output_file:
        
        # Create a minimal test dataset
        test_data = pd.DataFrame({
            "id": [1],
            "prompt": ["What is 2+2?"],
            "completion": ["The answer is 4."]
        })
        test_data.to_csv(input_file.name, index=False)
        
        # Create a test template directory
        template_dir = os.path.join(temp_dir, "templates")
        os.makedirs(template_dir)
        
        # Create keyword files
        with open(os.path.join(template_dir, "positive_keywords.txt"), "w") as f:
            f.write("correct\naccurate\nanswer\n4")
        
        with open(os.path.join(template_dir, "negative_keywords.txt"), "w") as f:
            f.write("wrong\nincorrect\nfalse")
        
        # Create a minimal configuration
        config = {
            "input_csv": input_file.name,
            "output_csv": output_file.name,
            "template_dir": template_dir
        }
        
        try:
            # Run judge
            result = run_from_config(config)
            
            # Verify basic results
            assert result["success"] is True
            assert config["output_csv"] in result["artifacts"]
            
            # Verify output file was created
            assert os.path.exists(output_file.name)
            
            # Check output data
            output_data = pd.read_csv(output_file.name)
            assert len(output_data) == 1
            assert "judge_rating" in output_data.columns
            assert "judge_justification" in output_data.columns
            
            # Since our completion has "4" and "answer" which are positive keywords,
            # the score should be high (5)
            assert output_data["judge_rating"].iloc[0] == 5
            
            # Check justification contains the positive hits
            justification = json.loads(output_data["judge_justification"].iloc[0])
            assert "pos_hits" in justification
            assert "answer" in justification["pos_hits"]
            assert "4" in justification["pos_hits"]
        
        finally:
            # Clean up temporary files
            os.unlink(input_file.name)
            if os.path.exists(output_file.name):
                os.unlink(output_file.name)