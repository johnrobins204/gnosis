from abc import ABC, abstractmethod

class Metric(ABC):
    """Abstract base class for all metrics."""

    @abstractmethod
    def calculate(self, data):
        """Calculate the metric based on the provided data."""
        pass

    @abstractmethod
    def name(self):
        """Return the name of the metric."""
        pass

class MetricRegistry:
    """Registry for dynamically loading metrics."""
    _registry = {}

    @classmethod
    def register(cls, metric_class):
        """Register a metric class."""
        cls._registry[metric_class.__name__] = metric_class

    @classmethod
    def get_metric(cls, name):
        """Retrieve a metric class by name."""
        if name not in cls._registry:
            raise ValueError(f"Metric '{name}' is not registered.")
        return cls._registry.get(name)

    @classmethod
    def list_metrics(cls):
        """List all registered metrics."""
        return list(cls._registry.keys())