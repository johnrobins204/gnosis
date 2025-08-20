import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import json
import os
import tempfile

from src.judge import run_from_config, _load_keywords, _score_completion

@pytest.fixture
def sample_config():
    """Sample judge configuration."""
    return {
        "input_csv": "completions.csv",
        "output_csv": "ratings.csv",
        "template_dir": "templates/default"
    }

@pytest.fixture
def sample_completions():
    """Sample completions data for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "prompt": [
            "What is recursion?",
            "Explain machine learning.",
            "Define artificial intelligence."
        ],
        "completion": [
            "Recursion is when a function calls itself.",
            "Machine learning is a subset of AI that enables systems to learn from data.",
            "AI is the simulation of human intelligence in machines."
        ],
        "model": ["Llama:8B", "Llama:8B", "Llama:8B"]
    })

@pytest.fixture
def sample_keywords():
    """Sample positive and negative keywords."""
    return {
        "positive": ["accurate", "correct", "helpful", "clear"],
        "negative": ["wrong", "incorrect", "confusing", "irrelevant"]
    }

def test_load_keywords():
    """Test loading keywords from template directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test template directory
        template_dir = os.path.join(temp_dir, "templates")
        os.makedirs(template_dir)
        
        # Create positive_keywords.txt
        with open(os.path.join(template_dir, "positive_keywords.txt"), "w") as f:
            f.write("accurate\ncorrect\nhelpful\nclear")
        
        # Create negative_keywords.txt
        with open(os.path.join(template_dir, "negative_keywords.txt"), "w") as f:
            f.write("wrong\nincorrect\nconfusing\nirrelevant")
        
        # Test loading keywords
        keywords = _load_keywords(template_dir)
        
        # Verify
        assert "positive" in keywords
        assert "negative" in keywords
        assert len(keywords["positive"]) == 4
        assert "helpful" in keywords["positive"]
        assert "incorrect" in keywords["negative"]

def test_load_keywords_missing_dir():
    """Test loading keywords when directory doesn't exist."""
    keywords = _load_keywords("nonexistent_dir")
    assert keywords == {"positive": [], "negative": []}

def test_score_completion():
    """Test scoring a completion based on keywords."""
    keywords = {
        "positive": ["recursion", "function", "calls itself"],
        "negative": ["loop", "iteration"]
    }
    
    # Test positive keywords only
    score1 = _score_completion("Recursion is when a function calls itself.", keywords)
    assert score1["rating"] == 5  # Max rating for positive only
    assert len(score1["justification"]["pos_hits"]) > 0
    assert len(score1["justification"]["neg_hits"]) == 0
    
    # Test negative keywords only
    score2 = _score_completion("This uses a loop for iteration.", keywords)
    assert score2["rating"] == 1  # Min rating for negative only
    assert len(score2["justification"]["pos_hits"]) == 0
    assert len(score2["justification"]["neg_hits"]) > 0
    
    # Test mix of positive and negative
    score3 = _score_completion("Recursion is similar to a loop but uses function calls.", keywords)
    assert 1 <= score3["rating"] <= 5  # Middle range for mixed
    assert len(score3["justification"]["pos_hits"]) > 0
    assert len(score3["justification"]["neg_hits"]) > 0
    
    # Test no keywords matched
    score4 = _score_completion("This is a short answer.", keywords)
    assert 1 <= score4["rating"] <= 5  # Should use length heuristic
    assert len(score4["justification"]["pos_hits"]) == 0
    assert len(score4["justification"]["neg_hits"]) == 0
    assert score4["justification"]["len_words"] == 5

@patch("src.judge.load_csv")
@patch("src.judge.write_dataframe")
@patch("src.judge._load_keywords")
def test_run_from_config_basic(mock_load_keywords, mock_write, mock_load, 
                              sample_config, sample_completions, sample_keywords):
    """Test basic functionality of run_from_config."""
    # Setup mocks
    mock_load.return_value = sample_completions
    mock_load_keywords.return_value = sample_keywords
    
    # Run the function
    result = run_from_config(sample_config)
    
    # Verify calls
    mock_load.assert_called_once_with(sample_config["input_csv"])
    mock_load_keywords.assert_called_once_with(sample_config["template_dir"])
    mock_write.assert_called_once()
    
    # Verify result structure
    assert result["success"] is True
    assert sample_config["output_csv"] in result["artifacts"]
    
    # Check the dataframe that was written
    write_df_arg = mock_write.call_args[0][0]  # First arg to write_dataframe
    
    # Should have the original columns plus judge columns
    assert "id" in write_df_arg.columns
    assert "prompt" in write_df_arg.columns
    assert "completion" in write_df_arg.columns
    assert "judge_rating" in write_df_arg.columns
    assert "judge_justification" in write_df_arg.columns

@patch("src.judge.load_csv")
@patch("src.judge.write_dataframe")
@patch("src.judge._load_keywords")
def test_column_name_flexibility(mock_load_keywords, mock_write, mock_load, sample_config, sample_keywords):
    """Test flexibility in completion column names."""
    # Test with "response" column instead of "completion"
    response_df = pd.DataFrame({
        "id": [1],
        "prompt": ["What is recursion?"],
        "response": ["Recursion is when a function calls itself."]
    })
    
    # Test with "answer" column
    answer_df = pd.DataFrame({
        "id": [1],
        "prompt": ["What is recursion?"],
        "answer": ["Recursion is when a function calls itself."]
    })
    
    mock_load_keywords.return_value = sample_keywords
    
    # Test "response" column
    mock_load.return_value = response_df
    result1 = run_from_config(sample_config)
    assert result1["success"] is True
    
    # Test "answer" column
    mock_load.return_value = answer_df
    result2 = run_from_config(sample_config)
    assert result2["success"] is True
    
    # Verify both cases produced ratings
    assert "judge_rating" in mock_write.call_args_list[0][0][0].columns
    assert "judge_rating" in mock_write.call_args_list[1][0][0].columns

@patch("src.judge.load_csv")
@patch("src.judge.write_dataframe")
def test_run_from_config_missing_keys(mock_write, mock_load):
    """Test handling of missing config keys."""
    # Test missing input_csv
    result = run_from_config({"output_csv": "output.csv", "template_dir": "templates"})
    assert result["success"] is False
    assert "missing config key: input_csv" in result["error"]
    
    # Test missing output_csv
    result = run_from_config({"input_csv": "input.csv", "template_dir": "templates"})
    assert result["success"] is False
    assert "missing config key: output_csv" in result["error"]
    
    # Test missing template_dir
    result = run_from_config({"input_csv": "input.csv", "output_csv": "output.csv"})
    assert result["success"] is False
    assert "missing config key: template_dir" in result["error"]

@patch("src.judge.load_csv")
def test_run_from_config_load_error(mock_load):
    """Test handling of errors when loading input file."""
    # Setup mock to raise exception
    mock_load.side_effect = Exception("File not found")
    
    result = run_from_config({
        "input_csv": "nonexistent.csv",
        "output_csv": "output.csv",
        "template_dir": "templates"
    })
    
    assert result["success"] is False
    assert "failed to load input_csv" in result["error"]