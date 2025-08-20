import pytest
import pandas as pd
import numpy as np
from src.analytics.metrics.aggregate_metrics import (
    VarianceExplainedMetric, 
    EffectSizeAggregateMetric,
    ReliabilityMetric
)

@pytest.fixture
def variance_data():
    """Create sample data for variance explained tests."""
    np.random.seed(42)
    data = []
    
    # Group A: values around 5
    for i in range(20):
        data.append({"group": "A", "value": np.random.normal(5, 1)})
    
    # Group B: values around 8
    for i in range(20):
        data.append({"group": "B", "value": np.random.normal(8, 1)})
    
    # Group C: values around 3
    for i in range(20):
        data.append({"group": "C", "value": np.random.normal(3, 1)})
    
    return pd.DataFrame(data)

@pytest.fixture
def effect_size_data():
    """Create sample data for effect size tests."""
    np.random.seed(42)
    data = []
    
    # Group A: values around 10
    for i in range(30):
        data.append({"group": "A", "value": np.random.normal(10, 2)})
    
    # Group B: values around 15
    for i in range(30):
        data.append({"group": "B", "value": np.random.normal(15, 2)})
    
    return pd.DataFrame(data)

@pytest.fixture
def reliability_data():
    """Create sample data for reliability tests."""
    np.random.seed(42)
    data = []
    
    # 10 items rated by 5 observers
    for item in range(10):
        # Base score for this item
        base_score = np.random.normal(7, 1)
        
        for observer in range(5):
            # Observer bias
            observer_bias = np.random.normal(0, 0.5)
            
            # Random noise
            noise = np.random.normal(0, 0.3)
            
            # Final rating
            rating = base_score + observer_bias + noise
            
            # Ensure ratings are within reasonable range (1-10)
            rating = max(1, min(10, rating))
            
            data.append({
                "item_id": f"item_{item}",
                "observer_id": f"observer_{observer}",
                "rating": rating
            })
    
    return pd.DataFrame(data)

def test_variance_explained_metric(variance_data):
    """Test the variance explained metric."""
    metric = VarianceExplainedMetric(factor_column="group", value_column="value")
    
    # Calculate variance explained
    result = metric.calculate(variance_data)
    
    # Verify results
    assert "variance_explained" in result
    assert "total_variance" in result
    assert "within_group_variance" in result
    
    # With our test data, groups should explain a large portion of variance
    assert 0.5 <= result["variance_explained"] <= 1.0
    
    # Test with missing columns
    with pytest.raises(ValueError):
        bad_metric = VarianceExplainedMetric(factor_column="missing", value_column="value")
        bad_metric.calculate(variance_data)

def test_effect_size_metric(effect_size_data):
    """Test the effect size metric."""
    metric = EffectSizeAggregateMetric(group_column="group", value_column="value")
    
    # Calculate effect sizes
    result = metric.calculate(effect_size_data)
    
    # There should be one comparison: A vs B
    assert "A_vs_B" in result
    
    # Check effect size metrics
    effect_size = result["A_vs_B"]
    assert "cohen_d" in effect_size
    assert "hedges_g" in effect_size
    assert "t_statistic" in effect_size
    assert "p_value" in effect_size
    assert "significant" in effect_size
    
    # With our test data, there should be a large effect size
    assert abs(effect_size["cohen_d"]) > 1.0
    
    # The result should be statistically significant
    assert effect_size["significant"] == True
    
    # Test with missing columns
    with pytest.raises(ValueError):
        bad_metric = EffectSizeAggregateMetric(group_column="missing", value_column="value")
        bad_metric.calculate(effect_size_data)

def test_reliability_metric(reliability_data):
    """Test the reliability metric."""
    metric = ReliabilityMetric(
        item_column="item_id", 
        observer_column="observer_id", 
        rating_column="rating"
    )
    
    # Calculate reliability
    result = metric.calculate(reliability_data)
    
    # Check reliability metrics
    assert "icc1" in result  # One-way random effects
    assert "icc2" in result  # Two-way random effects
    assert "icc3" in result  # Two-way mixed effects
    assert "icc1_k" in result  # Average measures, one-way random
    assert "icc2_k" in result  # Average measures, two-way random
    assert "icc3_k" in result  # Average measures, two-way mixed
    
    # ICC values should be between -1 and 1
    assert -1 <= result["icc1"] <= 1
    assert -1 <= result["icc2"] <= 1
    assert -1 <= result["icc3"] <= 1
    
    # Average measures should be higher than single measures
    assert result["icc1_k"] >= result["icc1"]
    assert result["icc2_k"] >= result["icc2"]
    assert result["icc3_k"] >= result["icc3"]
    
    # Test with missing columns
    with pytest.raises(ValueError):
        bad_metric = ReliabilityMetric(
            item_column="missing", 
            observer_column="observer_id", 
            rating_column="rating"
        )
        bad_metric.calculate(reliability_data)