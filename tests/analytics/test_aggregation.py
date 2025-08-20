import pytest
import pandas as pd
import numpy as np
from src.analytics.aggregation import DataAggregator
from src.analytics.metrics.aggregate_metrics import (
    VarianceExplainedMetric, 
    EffectSizeAggregateMetric,
    ReliabilityMetric
)

@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    np.random.seed(42)
    # Create the appropriate means for each group
    means = [7] * 15 + [8] * 15 + [6] * 15 + [9] * 15
    
    data = {
        "experiment_id": ["exp1"] * 30 + ["exp2"] * 30,
        "response_id": [f"r{i//3}" for i in range(60)],
        "prompt_id": ["p1"] * 15 + ["p2"] * 15 + ["p1"] * 15 + ["p2"] * 15,
        "model": ["model1"] * 30 + ["model2"] * 30,
        "response_text": [f"Sample response {i}" for i in range(60)],
        "score": [np.random.normal(loc=means[i], scale=1) for i in range(60)]
    }
    return pd.DataFrame(data)

@pytest.fixture
def reliability_data():
    """Create a dataset for testing reliability metrics."""
    np.random.seed(42)
    data = []
    for item_id in range(20):  # 20 items
        for observer_id in range(5):  # 5 observers
            # Base score depends on the item
            base_score = np.random.normal(loc=7, scale=1.5)
            # Add observer bias and random noise
            observer_bias = np.random.normal(loc=0, scale=0.5)
            noise = np.random.normal(loc=0, scale=0.3)
            score = base_score + observer_bias + noise
            data.append({
                "item_id": f"item_{item_id}",
                "observer_id": f"observer_{observer_id}",
                "rating": min(max(score, 1), 10)  # Clamp to 1-10 scale
            })
    return pd.DataFrame(data)

def test_data_aggregator_init():
    """Test DataAggregator initialization."""
    aggregator = DataAggregator()
    assert "mean" in aggregator.aggregation_functions
    assert "median" in aggregator.aggregation_functions
    assert "std" in aggregator.aggregation_functions
    assert callable(aggregator.aggregation_functions["mean"])

def test_register_aggregation_function():
    """Test registering a custom aggregation function."""
    aggregator = DataAggregator()
    def custom_func(x):
        return np.percentile(x, 75)
    
    aggregator.register_aggregation_function("p75", custom_func)
    assert "p75" in aggregator.aggregation_functions
    assert aggregator.aggregation_functions["p75"] is custom_func

def test_aggregate(sample_data):
    """Test basic aggregation functionality."""
    aggregator = DataAggregator()
    
    # Aggregate by experiment and prompt
    result = aggregator.aggregate(
        sample_data,
        group_by=["experiment_id", "prompt_id"],
        metrics={"avg_score": "mean", "score_std": "std", "count": "count"}
    )
    
    assert result.shape == (4, 5)  # 4 groups, 5 columns (2 group cols + 3 metrics)
    assert set(result.columns) == {"experiment_id", "prompt_id", "avg_score", "score_std", "count"}
    assert result["count"].sum() == 60  # All rows accounted for

def test_multi_level_pipeline(sample_data):
    """Test the multi-level pipeline functionality."""
    aggregator = DataAggregator()
    
    # Define a pipeline with multiple stages
    pipeline_config = [
        {
            "name": "raw_metrics",
            "metrics": {
                "response_length": lambda row: len(row["response_text"]),
                "is_high_score": lambda row: 1 if row["score"] > 7.5 else 0
            },
            "output_to_next": True
        },
        {
            "name": "response_aggregation",
            "group_by": ["response_id", "experiment_id", "model"],
            "metrics": {
                "avg_score": "mean",
                "high_score_freq": "mean",
                "response_count": "count"
            },
            "output_to_next": True
        },
        {
            "name": "experiment_summary",
            "group_by": ["experiment_id", "model"],
            "metrics": {
                "mean_score": "mean",
                "std_score": "std",
                "high_score_rate": "mean"
            }
        }
    ]
    
    results = aggregator.multi_level_pipeline(sample_data, pipeline_config)
    
    # Check if all stages are present
    assert set(results.keys()) == {"raw_metrics", "response_aggregation", "experiment_summary"}
    
    # Check raw metrics
    assert "response_length" in results["raw_metrics"].columns
    assert "is_high_score" in results["raw_metrics"].columns
    
    # Check response aggregation
    assert "avg_score" in results["response_aggregation"].columns
    assert "high_score_freq" in results["response_aggregation"].columns
    assert results["response_aggregation"].shape[0] == 20  # 20 unique responses
    
    # Check experiment summary
    assert "mean_score" in results["experiment_summary"].columns
    assert "high_score_rate" in results["experiment_summary"].columns
    assert results["experiment_summary"].shape[0] == 2  # 2 experiments

def test_empty_data():
    """Test handling of empty dataframes."""
    aggregator = DataAggregator()
    empty_df = pd.DataFrame(columns=["col1", "col2"])
    
    # Test aggregation with empty data
    result = aggregator.aggregate(
        empty_df,
        group_by=["col1"],
        metrics={"mean_col2": "mean"}
    )
    assert result.empty
    
    # Test pipeline with empty data
    pipeline_config = [
        {
            "name": "stage1",
            "group_by": ["col1"],
            "metrics": {"mean_col2": "mean"}
        }
    ]
    results = aggregator.multi_level_pipeline(empty_df, pipeline_config)
    assert results["stage1"].empty

def test_filter_in_pipeline(sample_data):
    """Test filtering within the pipeline."""
    aggregator = DataAggregator()
    
    # Define a pipeline with filtering
    pipeline_config = [
        {
            "name": "filtered_data",
            "filter": lambda row: row["score"] > 7.0,
            "metrics": {
                "high_enough": lambda row: 1
            }
        }
    ]
    
    results = aggregator.multi_level_pipeline(sample_data, pipeline_config)
    
    # The filtered data should only contain rows with score > 7.0
    assert len(results["filtered_data"]) < len(sample_data)
    assert all(results["filtered_data"]["score"] > 7.0)

def test_transform_in_pipeline(sample_data):
    """Test transformation within the pipeline."""
    aggregator = DataAggregator()
    
    # Define a pipeline with transformation
    pipeline_config = [
        {
            "name": "transformed_data",
            "transform": lambda df: df.assign(score_normalized=(df["score"] - df["score"].mean()) / df["score"].std()),
            "metrics": {}
        }
    ]
    
    results = aggregator.multi_level_pipeline(sample_data, pipeline_config)
    
    # The transformed data should contain the new column
    assert "score_normalized" in results["transformed_data"].columns
    assert abs(results["transformed_data"]["score_normalized"].mean()) < 1e-10  # Mean should be very close to 0

def test_error_handling(sample_data):
    """Test error handling for invalid inputs."""
    aggregator = DataAggregator()
    
    # Test invalid group-by column
    with pytest.raises(ValueError):
        aggregator.aggregate(
            sample_data,
            group_by=["nonexistent_column"],
            metrics={"avg_score": "mean"}
        )
    
    # Test invalid aggregation function
    with pytest.raises(ValueError):
        aggregator.aggregate(
            sample_data,
            group_by=["experiment_id"],
            metrics={"invalid_metric": "nonexistent_function"}
        )