import pytest
from src.analytics.metrics.statistical import ConfidenceIntervalMetric, EffectSizeMetric, HypothesisTestingMetric

def test_confidence_interval_metric():
    metric = ConfidenceIntervalMetric()
    data = {"values": [1, 2, 3, 4, 5]}
    result = metric.calculate(data)
    assert "mean" in result
    assert result["lower"] < result["mean"] < result["upper"]

def test_effect_size_metric():
    metric = EffectSizeMetric()
    data = {"group1": [1, 2, 3], "group2": [4, 5, 6]}
    result = metric.calculate(data)
    assert "cohen_d" in result

def test_hypothesis_testing_metric():
    metric = HypothesisTestingMetric()
    data = {"group1": [1, 2, 3], "group2": [4, 5, 6]}
    result = metric.calculate(data)
    assert "p_value" in result