import pytest
import pandas as pd
import numpy as np
from src.analytics.aggregation import DataAggregator

@pytest.mark.parametrize("config", [
    {"group_by": ["experiment_id"], "metric": "mean"},
    {"group_by": ["experiment_id", "model"], "metric": "median"},
    {"group_by": ["response_id"], "metric": "std"},
    {"group_by": ["experiment_id", "prompt_id"], "metric": "count"},
])
def test_config_variations(config):
    """Test DataAggregator with different configuration settings."""
    # Create test data
    test_data = pd.DataFrame({
        "experiment_id": ["exp1", "exp1", "exp2", "exp2"],
        "model": ["model1", "model1", "model2", "model2"],
        "response_id": ["r1", "r2", "r3", "r4"],
        "prompt_id": ["p1", "p2", "p1", "p2"],
        "score": [7.5, 8.2, 6.4, 9.1]
    })
    
    # Create aggregator
    aggregator = DataAggregator()
    
    # Set up metrics dict based on config
    metrics = {f"{config['metric']}_score": config['metric']}
    
    # Perform aggregation
    result = aggregator.aggregate(
        data=test_data,
        group_by=config["group_by"],
        metrics=metrics
    )
    
    # Verify basic properties
    assert not result.empty
    assert len(result) <= len(test_data)  # Aggregation should reduce or maintain row count
    assert all(col in result.columns for col in config["group_by"])  # Group columns preserved
    assert f"{config['metric']}_score" in result.columns  # Metric column created