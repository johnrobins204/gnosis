# Analytics Addin/Plugin System

This document describes how to develop, register, and test analytics addins (plugins) for the gnosis analytics framework.

## Overview

The analytics addin system allows you to extend gnosis with custom metrics, aggregation functions, or workflow steps. Addins are Python classes or functions that follow a defined interface and are registered with the analytics registry.

## Writing an Addin

1. **Create a new Python file in `src/analytics/metrics/` or your own module.**
2. **Inherit from the appropriate base class:**
   - For metrics: inherit from `BaseMetric` (see `metrics/base.py`).
   - For aggregations: follow the callable signature expected by `DataAggregator`.
3. **Implement required methods and docstrings.**

Example metric addin:
```python
from src.analytics.metrics.base import BaseMetric

class MyCustomMetric(BaseMetric):
    """A custom metric that counts unique values."""
    name = "unique_count"
    description = "Counts the number of unique values in a column."

    def compute(self, data, **kwargs):
        return data.nunique()
```

## Registering an Addin

To make your addin discoverable:
- Use the `MetricRegistry.register()` method for metrics.
- For aggregations, use `DataAggregator.register_aggregation_function()`.

Example:
```python
from src.analytics.registry import MetricRegistry
from .my_custom_metric import MyCustomMetric

MetricRegistry.register(MyCustomMetric)
```

## Testing Addins

- Add tests in `tests/analytics/` for your addin.
- Use pytest to validate compute logic and registration.

Example test:
```python
def test_unique_count_metric():
    from src.analytics.metrics.my_custom_metric import MyCustomMetric
    import pandas as pd
    metric = MyCustomMetric()
    df = pd.DataFrame({'a': [1, 2, 2, 3]})
    assert metric.compute(df['a']) == 3
```

## Listing and Using Addins

- List all registered metrics via `MetricRegistry.list_metrics()`.
- Use addins in analytics pipelines by specifying their name in config or code.

## Best Practices
- Add docstrings and metadata (name, description) to your addin.
- Write tests for all new addins.
- Follow the interface of the base class or registry.

---
For more details, see `src/analytics/metrics/base.py`, `src/analytics/registry.py`, and the analytics API docs.
