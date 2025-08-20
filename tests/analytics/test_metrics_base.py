import pytest
from src.analytics.metrics.base import Metric, MetricRegistry

class DummyMetric(Metric):
    def calculate(self, data):
        return {"dummy": 1}

    def name(self):
        return "DummyMetric"

def test_metric_registration():
    MetricRegistry.register(DummyMetric)
    assert "DummyMetric" in MetricRegistry.list_metrics()

def test_metric_retrieval():
    metric_class = MetricRegistry.get_metric("DummyMetric")
    assert metric_class is DummyMetric

def test_metric_calculation():
    metric = DummyMetric()
    result = metric.calculate({})
    assert result == {"dummy": 1}