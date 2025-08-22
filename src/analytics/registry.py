# Handles metric registration and lookup

class MetricRegistry:
    _metrics = {}

    @classmethod
    def register(cls, metric_cls):
        cls._metrics[metric_cls.__name__] = metric_cls

    @classmethod
    def get_metrics(cls, metric_names=None):
        if metric_names is None:
            return list(cls._metrics.values())
        return [cls._metrics[name] for name in metric_names if name in cls._metrics]