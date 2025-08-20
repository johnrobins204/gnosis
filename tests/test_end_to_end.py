import pytest
import pandas as pd
import os
import tempfile
from unittest.mock import patch, MagicMock

from src.inference import run_from_config as run_inference
from src.judge import run_from_config as run_judge
from src.analytics import run_from_config as run_analytics
from src.types import ModelResponse

@pytest.fixture
def test_directory():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create template directories
        judge_template_dir = os.path.join(temp_dir, "judge_templates")
        os.makedirs(judge_template_dir)
        
        # Create keyword files for judge
        with open(os.path.join(judge_template_dir, "positive_keywords.txt"), "w") as f:
            f.write("correct\naccurate\nanswer\nfour\n4")
        
        with open(os.path.join(judge_template_dir, "negative_keywords.txt"), "w") as f:
            f.write("wrong\nincorrect\nfalse")
        
        yield temp_dir

def test_full_workflow(test_directory):
    """
    Test the complete workflow: Inference → Judge → Analytics
    """
    # File paths
    prompts_csv = os.path.join(test_directory, "prompts.csv")
    completions_csv = os.path.join(test_directory, "completions.csv")
    ratings_csv = os.path.join(test_directory, "ratings.csv")
    analysis_csv = os.path.join(test_directory, "analysis.csv")
    judge_template_dir = os.path.join(test_directory, "judge_templates")
    
    # Create initial prompts data with different models
    test_prompts = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "prompt": ["What is 2+2?"] * 4,
        "model": ["Llama:8B", "Llama:8B", "gpt-3.5-turbo", "gpt-3.5-turbo"]
    })
    test_prompts.to_csv(prompts_csv, index=False)
    
    # Step 1: Run Inference
    inference_config = {
        "input_csv": prompts_csv,
        "output_csv": completions_csv,
        "default_model": "Llama:8B"
    }
    
    with patch("src.inference.get_model_instance") as mock_get_model:
        mock_model = mock_get_model.return_value
        
        # Each prompt gets its own response with different quality
        mock_model.generate.side_effect = [
            ModelResponse(
                model="Llama:8B",
                prompt="What is 2+2?",
                completion="The answer is 4.",
                metadata={"tokens": 5}
            ),
            ModelResponse(
                model="Llama:8B",
                prompt="What is 2+2?",
                completion="It's 4.",
                metadata={"tokens": 3}
            ),
            ModelResponse(
                model="gpt-3.5-turbo",
                prompt="What is 2+2?",
                completion="2+2 equals 4.",
                metadata={"tokens": 4}
            ),
            ModelResponse(
                model="gpt-3.5-turbo",
                prompt="What is 2+2?",
                completion="Four.",
                metadata={"tokens": 1}
            )
        ]
        
        # Run inference
        inference_result = run_inference(inference_config)
        
        # Verify inference succeeded
        assert inference_result["success"] is True
        assert os.path.exists(completions_csv)
    
    # Step 2: Run Judge
    judge_config = {
        "input_csv": completions_csv,
        "output_csv": ratings_csv,
        "template_dir": judge_template_dir
    }
    
    # Run judge
    judge_result = run_judge(judge_config)
    
    # Verify judge succeeded
    assert judge_result["success"] is True
    assert os.path.exists(ratings_csv)
    
    # Check judge output to make sure ratings were generated
    ratings_df = pd.read_csv(ratings_csv)
    assert len(ratings_df) == 4
    assert "judge_rating" in ratings_df.columns
    
    # Step 3: Run Analytics
    analytics_config = {
        "input_csv": ratings_csv,
        "output_csv": analysis_csv,
        "group_by": ["model"]
    }
    
    # Run analytics
    analytics_result = run_analytics(analytics_config)
    
    # Verify analytics succeeded
    assert analytics_result["success"] is True
    assert os.path.exists(analysis_csv)
    
    # Check analytics output
    analysis_df = pd.read_csv(analysis_csv)
    assert len(analysis_df) == 2  # Two models
    assert "model" in analysis_df.columns
    assert "count" in analysis_df.columns
    assert "avg_judge_rating" in analysis_df.columns
    
    # Verify counts are correct
    assert analysis_df[analysis_df["model"] == "Llama:8B"]["count"].iloc[0] == 2
    assert analysis_df[analysis_df["model"] == "gpt-3.5-turbo"]["count"].iloc[0] == 2