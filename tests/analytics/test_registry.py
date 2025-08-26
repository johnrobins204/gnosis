import pytest
from src.analytics.metrics.base import Metric, MetricRegistry

class DummyMetric(Metric):
    def calculate(self, data):
        return 42
    def name(self):
        return "dummy"

def test_metric_registration_and_lookup():
    MetricRegistry.register(DummyMetric)
    assert "DummyMetric" in MetricRegistry.list_metrics()
    metric_cls = MetricRegistry.get_metric("DummyMetric")
    assert metric_cls is not None
    assert issubclass(metric_cls, Metric)
    metric = metric_cls()
    assert metric.calculate(None) == 42
    assert metric.name() == "dummy"
