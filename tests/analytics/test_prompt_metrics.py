import pytest
from src.analytics.metrics.prompt_engineering import TokenUtilizationMetric, InstructionAdherenceMetric, PerformanceCostRatioMetric

def test_token_utilization_metric():
    metric = TokenUtilizationMetric()
    data = {"prompts": ["Test prompt"], "responses": ["Test response"]}
    result = metric.calculate(data)
    assert "utilization" in result

def test_instruction_adherence_metric():
    metric = InstructionAdherenceMetric()
    data = {"instructions": ["Follow this"], "responses": ["Follow this"]}
    result = metric.calculate(data)
    assert result["average_adherence"] == 1.0

def test_performance_cost_ratio_metric():
    metric = PerformanceCostRatioMetric()
    data = {"performance_scores": [1, 2], "costs": [1, 2]}
    result = metric.calculate(data)
    assert "performance_cost_ratios" in result