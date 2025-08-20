import pytest
import pandas as pd
import numpy as np
from src.analytics.aggregation import DataAggregator
from src.analytics.metrics.base import MetricRegistry
from src.analytics.metrics.nlp import BLEUMetric

def test_analytics_pipeline_smoke():
    """
    Smoke test to verify the entire analytics pipeline works.
    This test exercises multiple components to ensure they work together.
    """
    # 1. Create test data that simulates experiment results
    experiment_data = pd.DataFrame({
        "experiment_id": ["exp1"] * 6 + ["exp2"] * 6,
        "model": ["model1"] * 3 + ["model2"] * 3 + ["model1"] * 3 + ["model2"] * 3,
        "response_id": [f"r{i//3 + 1}" for i in range(12)],
        "prompt_id": ["p1", "p1", "p1", "p2", "p2", "p2"] * 2,
        "response_text": [f"This is response {i}" for i in range(12)],
        "reference_text": ["This is reference text"] * 12,
        "score": np.random.normal(loc=7.5, scale=1.5, size=12).tolist()
    })
    
    # 2. Initialize the DataAggregator
    aggregator = DataAggregator()
    
    # 3. Register a custom aggregation function
    def percentile_75(x):
        return np.percentile(x, 75)
    
    aggregator.register_aggregation_function("p75", percentile_75)
    
    # 4. Define a multi-level analysis pipeline
    pipeline_config = [
        # Stage 1: Calculate raw metrics
        {
            "name": "raw_metrics",
            "metrics": {
                "response_length": lambda row: len(row["response_text"]),
                "is_high_score": lambda row: 1 if row["score"] > 7.5 else 0
            },
            "output_to_next": True
        },
        # Stage 2: Aggregate to response level
        {
            "name": "response_level",
            "group_by": ["response_id", "experiment_id", "model"],
            "metrics": {
                "avg_score": "mean",
                "p75_score": "p75",  # Using our custom aggregation
                "observation_count": "count"
            },
            "output_to_next": True
        },
        # Stage 3: Final experiment-level aggregation
        {
            "name": "experiment_summary",
            "group_by": ["experiment_id", "model"],
            "metrics": {
                "mean_score": "mean",
                "score_variability": "std"
            }
        }
    ]
    
    # 5. Execute the pipeline
    results = aggregator.multi_level_pipeline(experiment_data, pipeline_config)
    
    # 6. Verify each stage produced expected outputs
    assert "raw_metrics" in results
    assert "response_level" in results
    assert "experiment_summary" in results
    
    # 7. Check raw metrics
    raw_df = results["raw_metrics"]
    assert "response_length" in raw_df.columns
    assert "is_high_score" in raw_df.columns
    assert len(raw_df) == len(experiment_data)
    
    # 8. Check response level aggregation
    resp_df = results["response_level"]
    assert "avg_score" in resp_df.columns
    assert "p75_score" in resp_df.columns
    assert "observation_count" in resp_df.columns
    assert len(resp_df) <= len(experiment_data)
    
    # 9. Check experiment summary
    exp_df = results["experiment_summary"]
    assert "mean_score" in exp_df.columns
    assert "score_variability" in exp_df.columns
    assert len(exp_df) == len(experiment_data["experiment_id"].unique()) * len(experiment_data["model"].unique())
    
    # 10. Test integration with metrics system
    bleu = BLEUMetric()
    bleu_data = {"references": experiment_data["reference_text"].tolist(), 
                 "hypotheses": experiment_data["response_text"].tolist()}
    bleu_result = bleu.calculate(bleu_data)
    
    # Verify BLEU score is valid
    assert isinstance(bleu_result, list)
    assert all(0 <= score <= 1 for score in bleu_result)

