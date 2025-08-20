import pytest
import pandas as pd
import numpy as np

from src.analytics.metrics import BasicMetrics

def test_basic_metrics_mean():
    """Test the mean metric function."""
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    
    # Create mean function for "value" column
    mean_func = BasicMetrics.mean("value")
    
    # Test function
    result = mean_func(df)
    assert result == 3.0
    
    # Test with missing column
    result = BasicMetrics.mean("nonexistent")(df)
    assert result is None

def test_basic_metrics_count():
    """Test the count metric function."""
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    
    # Create count function
    count_func = BasicMetrics.count()
    
    # Test function
    result = count_func(df)
    assert result == 5
    
    # Test with empty dataframe
    empty_df = pd.DataFrame()
    result = count_func(empty_df)
    assert result == 0

def test_basic_metrics_sum():
    """Test the sum metric function."""
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    
    # Create sum function
    sum_func = BasicMetrics.sum("value")
    
    # Test function
    result = sum_func(df)
    assert result == 15
    
    # Test with missing column
    result = BasicMetrics.sum("nonexistent")(df)
    assert result is None

def test_basic_metrics_min_max():
    """Test the min and max metric functions."""
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    
    # Test min
    min_func = BasicMetrics.min("value")
    assert min_func(df) == 1
    
    # Test max
    max_func = BasicMetrics.max("value")
    assert max_func(df) == 5
    
    # Test with missing column
    assert BasicMetrics.min("nonexistent")(df) is None
    assert BasicMetrics.max("nonexistent")(df) is None

def test_metrics_with_nulls():
    """Test how metrics handle null values."""
    df = pd.DataFrame({"value": [1, 2, None, 4, 5]})
    
    # Mean should ignore nulls
    mean_func = BasicMetrics.mean("value")
    assert mean_func(df) == 3.0
    
    # Sum should ignore nulls
    sum_func = BasicMetrics.sum("value")
    assert sum_func(df) == 12
    
    # Min/Max should ignore nulls
    min_func = BasicMetrics.min("value")
    assert min_func(df) == 1
    
    max_func = BasicMetrics.max("value")
    assert max_func(df) == 5