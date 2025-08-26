# Analytics Metrics API

## Abstract Base Class: Metric

All metrics must inherit from the `Metric` abstract base class and implement the following methods:

- `calculate(self, data)`: Calculate the metric based on the provided data.
- `name(self)`: Return the name of the metric.

Example:
```python
from src.analytics.metrics.base import Metric

class MyMetric(Metric):
    def calculate(self, data):
        # Implement metric calculation
        return ...
    def name(self):
        return "my_metric"
```


## Metric Registration & Discovery

Metrics must be registered with the `MetricRegistry` to be discoverable and usable by the analytics system.

- `MetricRegistry.register(metric_class, name=None)`: Register a metric class with an optional custom name.
- `MetricRegistry.get_metric(name)`: Retrieve a metric class by name.
- `MetricRegistry.list_metrics()`: List all registered metric names.
- `MetricRegistry.list_metrics_with_metadata()`: List all registered metrics with their name and description/docstring.

Example:
```python
from src.analytics.metrics.base import MetricRegistry, MyMetric

MetricRegistry.register(MyMetric)
print(MetricRegistry.list_metrics())
print(MetricRegistry.list_metrics_with_metadata())
```

## Usage Example
```python
# Register and use a metric
MetricRegistry.register(MyMetric)
metric_cls = MetricRegistry.get_metric("MyMetric")
metric = metric_cls()
result = metric.calculate(data)
```

## Best Practices
- Always inherit from `Metric` for new metrics.
- Register metrics at import time (e.g., at the bottom of the metric module).
- Use descriptive names for metrics.
- Document required data format for each metric.
