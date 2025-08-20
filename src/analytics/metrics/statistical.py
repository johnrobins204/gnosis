import numpy as np
from scipy.stats import ttest_ind, norm
from src.analytics.metrics.base import Metric, MetricRegistry

class ConfidenceIntervalMetric(Metric):
    """Calculate confidence intervals for a given dataset."""

    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level

    def calculate(self, data):
        """Calculate confidence intervals for a list of values."""
        if not data["values"]:
            raise ValueError("Values cannot be empty.")
        values = np.array(data["values"])
        mean = np.mean(values)
        std_err = np.std(values, ddof=1) / np.sqrt(len(values))
        z_score = norm.ppf(1 - (1 - self.confidence_level) / 2)
        margin_of_error = z_score * std_err
        return {"mean": mean, "lower": mean - margin_of_error, "upper": mean + margin_of_error}

    def name(self):
        return "ConfidenceInterval"

class EffectSizeMetric(Metric):
    """Calculate effect sizes (Cohen's d, Hedges' g)."""

    def calculate(self, data):
        """Calculate effect sizes for two datasets."""
        group1, group2 = np.array(data["group1"]), np.array(data["group2"])
        mean_diff = np.mean(group1) - np.mean(group2)
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + (len(group2) - 1) * np.var(group2, ddof=1)) / (len(group1) + len(group2) - 2))
        cohen_d = mean_diff / pooled_std
        hedges_g = cohen_d * (1 - (3 / (4 * (len(group1) + len(group2)) - 9)))
        return {"cohen_d": cohen_d, "hedges_g": hedges_g}

    def name(self):
        return "EffectSize"

class HypothesisTestingMetric(Metric):
    """Perform hypothesis testing (t-tests)."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def calculate(self, data):
        """Perform a t-test for two datasets."""
        group1, group2 = np.array(data["group1"]), np.array(data["group2"])
        t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
        return {"t_stat": t_stat, "p_value": p_value, "significant": p_value < self.alpha}

    def name(self):
        return "HypothesisTesting"

# Register metrics
MetricRegistry.register(ConfidenceIntervalMetric)
MetricRegistry.register(EffectSizeMetric)
MetricRegistry.register(HypothesisTestingMetric)