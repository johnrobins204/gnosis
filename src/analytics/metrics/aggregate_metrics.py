
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from scipy import stats  # type: ignore
from typing import List, Dict, Any
from typing import Optional

from analytics.metrics.base import Metric
from analytics.registry import MetricRegistry

class AggregateMetric(Metric):
    """Base class for metrics that operate on aggregated data."""

    aggregation_level: str

    def __init__(self, aggregation_level: str) -> None:
        """
        Initialize with the aggregation level this metric operates at.
        """
        self.aggregation_level = aggregation_level

    def requires_aggregation(self) -> bool:
        """Indicates that this metric operates on aggregated data."""
        return True

class VarianceExplainedMetric(AggregateMetric):
    """Calculate the percentage of variance explained by different factors."""

    factor_column: str
    value_column: str

    def __init__(self, factor_column: str, value_column: str, aggregation_level: str = "experiment") -> None:
        super().__init__(aggregation_level)
        self.factor_column = factor_column
        self.value_column = value_column

    def calculate(self, data: pd.DataFrame) -> Dict[str, float]:
        if self.factor_column not in data.columns:
            raise ValueError(f"Factor column '{self.factor_column}' not found in data")
        if self.value_column not in data.columns:
            raise ValueError(f"Value column '{self.value_column}' not found in data")

        total_variance: float = float(np.var(data[self.value_column], ddof=1))
        grouped = data.groupby(self.factor_column)
        within_group_variance: float = float(grouped[self.value_column].var().mean())
        variance_explained: float = (total_variance - within_group_variance) / total_variance

        return {
            "variance_explained": variance_explained,
            "total_variance": total_variance,
            "within_group_variance": within_group_variance
        }

    def name(self) -> str:
        return "VarianceExplained"

class EffectSizeAggregateMetric(AggregateMetric):
    """Calculate effect sizes between groups in aggregated data."""

    group_column: str
    value_column: str

    def __init__(self, group_column: str, value_column: str, aggregation_level: str = "experiment") -> None:
        super().__init__(aggregation_level)
        self.group_column = group_column
        self.value_column = value_column

    def calculate(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        if self.group_column not in data.columns:
            raise ValueError(f"Group column '{self.group_column}' not found in data")
        if self.value_column not in data.columns:
            raise ValueError(f"Value column '{self.value_column}' not found in data")

        groups = data[self.group_column].unique()
        results: Dict[str, Dict[str, Any]] = {}
        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                group1_data = data[data[self.group_column] == group1][self.value_column]
                group2_data = data[data[self.group_column] == group2][self.value_column]

                mean_diff: float = float(group1_data.mean() - group2_data.mean())
                pooled_std: float = float(np.sqrt(((len(group1_data) - 1) * group1_data.var(ddof=1) +
                                                  (len(group2_data) - 1) * group2_data.var(ddof=1)) /
                                                 (len(group1_data) + len(group2_data) - 2)))
                cohen_d: float = mean_diff / pooled_std if pooled_std > 0 else 0.0
                correction: float = 1 - 3 / (4 * (len(group1_data) + len(group2_data) - 2) - 1)
                hedges_g: float = cohen_d * correction
                t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)  # type: ignore
                results[f"{group1}_vs_{group2}"] = {
                    "cohen_d": cohen_d,
                    "hedges_g": hedges_g,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": float(p_value) < 0.05  # type: ignore
                }
        return results

    def name(self) -> str:
        return "EffectSizeAggregate"

class ReliabilityMetric(AggregateMetric):
    """Calculate reliability metrics across multiple observations of the same item."""

    item_column: str
    observer_column: str
    rating_column: str

    def __init__(self, item_column: str, observer_column: str, rating_column: str, aggregation_level: str = "experiment") -> None:
        super().__init__(aggregation_level)
        self.item_column = item_column
        self.observer_column = observer_column
        self.rating_column = rating_column

    def calculate(self, data: pd.DataFrame) -> Dict[str, float]:
        required_cols = [self.item_column, self.observer_column, self.rating_column]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")

        ratings_wide: pd.DataFrame = data.pivot_table(
            index=self.item_column,
            columns=self.observer_column,
            values=self.rating_column
        )
        n_observers: int = ratings_wide.shape[1]
        n_items: int = ratings_wide.shape[0]
        ratings_wide = ratings_wide.fillna(ratings_wide.mean(axis=1))

        ms_items: float = float(n_observers * ratings_wide.var(axis=1).sum() / (n_items - 1))
        ms_observers: float = float(n_items * ratings_wide.var(axis=0).sum() / (n_observers - 1))
        total_sum_sq: float = float(((ratings_wide - ratings_wide.mean().mean())**2).sum().sum())
        items_sum_sq: float = float(n_observers * ((ratings_wide.mean(axis=1) - ratings_wide.mean().mean())**2).sum())
        observers_sum_sq: float = float(n_items * ((ratings_wide.mean(axis=0) - ratings_wide.mean().mean())**2).sum())
        residual_sum_sq: float = total_sum_sq - items_sum_sq - observers_sum_sq
        df_residual: int = (n_items - 1) * (n_observers - 1)
        ms_residual: float = float(residual_sum_sq / df_residual)

        icc1: float = (ms_items - ms_residual) / (ms_items + (n_observers - 1) * ms_residual)
        icc2: float = (ms_items - ms_residual) / (ms_items + (ms_observers - ms_residual) / n_items + (n_observers - 1) * ms_residual)
        icc3: float = (ms_items - ms_residual) / (ms_items + (n_observers - 1) * ms_residual)
        icc1_k: float = (ms_items - ms_residual) / ms_items
        icc2_k: float = (ms_items - ms_residual) / (ms_items + (ms_observers - ms_residual) / n_items)
        icc3_k: float = (ms_items - ms_residual) / ms_items

        return {
            "icc1": icc1,
            "icc2": icc2,
            "icc3": icc3,
            "icc1_k": icc1_k,
            "icc2_k": icc2_k,
            "icc3_k": icc3_k,
            "ms_items": ms_items,
            "ms_observers": ms_observers,
            "ms_residual": ms_residual,
            "n_items": float(n_items),
            "n_observers": float(n_observers)
        }

    def name(self) -> str:
        return "Reliability"

# Register metrics
MetricRegistry.register(VarianceExplainedMetric)
MetricRegistry.register(EffectSizeAggregateMetric)
MetricRegistry.register(ReliabilityMetric)