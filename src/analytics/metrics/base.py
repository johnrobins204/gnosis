from abc import ABC, abstractmethod


from typing import Any

class Metric(ABC):
    """Abstract base class for all metrics."""

    @abstractmethod
    def calculate(self, data: Any) -> Any:
        """Calculate the metric based on the provided data."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the name of the metric."""
        pass

from typing import Type, Dict, Optional, List, Any

class MetricRegistry:
    """Registry for dynamically loading metrics."""
    _registry: Dict[str, Type[Metric]] = {}

    @classmethod
    def register(cls, metric_class: Type[Metric], name: Optional[str] = None) -> None:
        """Register a metric class with an optional custom name."""
        key = name if name else metric_class.__name__
        cls._registry[key] = metric_class

    @classmethod
    def get_metric(cls, name: str) -> Type[Metric]:
        """Retrieve a metric class by name."""
        if name not in cls._registry:
            raise ValueError(f"Metric '{name}' is not registered.")
        return cls._registry[name]

    @classmethod
    def list_metrics(cls) -> List[str]:
        """List all registered metrics."""
        return list(cls._registry.keys())

    @classmethod
    def list_metrics_with_metadata(cls) -> List[Dict[str, Any]]:
        """List all registered metrics with their metadata (name, description/docstring)."""
        result: List[Dict[str, Any]] = []
        for name, metric_cls in cls._registry.items():
            doc = metric_cls.__doc__ or ""
            result.append({"name": name, "description": doc.strip()})
        return result