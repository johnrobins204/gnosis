import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Any

from src.analytics.metrics.base import Metric
from src.analytics.registry import MetricRegistry

class AggregateMetric(Metric):
    """Base class for metrics that operate on aggregated data."""
    
    def __init__(self, aggregation_level: str):
        """
        Initialize with the aggregation level this metric operates at.
        
        Args:
            aggregation_level: The level at which this metric should be calculated
                              (e.g., "response", "experiment", "model")
        """
        self.aggregation_level = aggregation_level
    
    def requires_aggregation(self) -> bool:
        """Indicates that this metric operates on aggregated data."""
        return True

class VarianceExplainedMetric(AggregateMetric):
    """Calculate the percentage of variance explained by different factors."""
    
    def __init__(self, factor_column: str, value_column: str, aggregation_level: str = "experiment"):
        """
        Initialize with the factor and value columns.
        
        Args:
            factor_column: Column containing the factor to analyze
            value_column: Column containing the numerical values
            aggregation_level: The level at which this metric should be calculated
        """
        super().__init__(aggregation_level)
        self.factor_column = factor_column
        self.value_column = value_column
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate the percentage of variance explained by the factor.
        
        Args:
            data: DataFrame containing the aggregated data
            
        Returns:
            Dictionary with variance explained statistics
        """
        # Verify columns exist
        if self.factor_column not in data.columns:
            raise ValueError(f"Factor column '{self.factor_column}' not found in data")
        if self.value_column not in data.columns:
            raise ValueError(f"Value column '{self.value_column}' not found in data")
        
        # Calculate total variance
        total_variance = np.var(data[self.value_column], ddof=1)
        
        # Calculate within-group variance
        grouped = data.groupby(self.factor_column)
        within_group_variance = grouped[self.value_column].var().mean()
        
        # Percentage of variance explained
        variance_explained = (total_variance - within_group_variance) / total_variance
        
        return {
            "variance_explained": variance_explained,
            "total_variance": total_variance,
            "within_group_variance": within_group_variance
        }
    
    def name(self):
        return "VarianceExplained"

class EffectSizeAggregateMetric(AggregateMetric):
    """Calculate effect sizes between groups in aggregated data."""
    
    def __init__(self, group_column: str, value_column: str, aggregation_level: str = "experiment"):
        """
        Initialize with the group and value columns.
        
        Args:
            group_column: Column containing the group labels
            value_column: Column containing the numerical values
            aggregation_level: The level at which this metric should be calculated
        """
        super().__init__(aggregation_level)
        self.group_column = group_column
        self.value_column = value_column
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate effect sizes between all pairs of groups.
        
        Args:
            data: DataFrame containing the aggregated data
            
        Returns:
            Dictionary mapping pairs of groups to effect size statistics
        """
        # Verify columns exist
        if self.group_column not in data.columns:
            raise ValueError(f"Group column '{self.group_column}' not found in data")
        if self.value_column not in data.columns:
            raise ValueError(f"Value column '{self.value_column}' not found in data")
        
        # Get unique groups
        groups = data[self.group_column].unique()
        
        # Calculate effect sizes for all pairs
        results = {}
        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                group1_data = data[data[self.group_column] == group1][self.value_column]
                group2_data = data[data[self.group_column] == group2][self.value_column]
                
                # Cohen's d
                mean_diff = group1_data.mean() - group2_data.mean()
                pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var(ddof=1) + 
                                     (len(group2_data) - 1) * group2_data.var(ddof=1)) / 
                                    (len(group1_data) + len(group2_data) - 2))
                
                cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                # Hedges' g (bias correction for small samples)
                correction = 1 - 3 / (4 * (len(group1_data) + len(group2_data) - 2) - 1)
                hedges_g = cohen_d * correction
                
                # T-test
                t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
                
                results[f"{group1}_vs_{group2}"] = {
                    "cohen_d": cohen_d,
                    "hedges_g": hedges_g,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
        
        return results
    
    def name(self):
        return "EffectSizeAggregate"

class ReliabilityMetric(AggregateMetric):
    """Calculate reliability metrics across multiple observations of the same item."""
    
    def __init__(self, item_column: str, observer_column: str, rating_column: str, 
                 aggregation_level: str = "experiment"):
        """
        Initialize with columns for reliability analysis.
        
        Args:
            item_column: Column identifying the items being rated
            observer_column: Column identifying the observers/raters
            rating_column: Column containing the ratings
            aggregation_level: The level at which this metric should be calculated
        """
        super().__init__(aggregation_level)
        self.item_column = item_column
        self.observer_column = observer_column
        self.rating_column = rating_column
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate reliability metrics (variance components, ICC).
        
        Args:
            data: DataFrame containing the rating data
            
        Returns:
            Dictionary with reliability statistics
        """
        # Verify columns exist
        required_cols = [self.item_column, self.observer_column, self.rating_column]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # Convert to wide format for ICC calculation
        ratings_wide = data.pivot_table(
            index=self.item_column,
            columns=self.observer_column,
            values=self.rating_column
        )
        
        # Basic consistency check - number of observations per item should be consistent
        n_observers = ratings_wide.shape[1]
        n_items = ratings_wide.shape[0]
        
        # Handle missing values by filling with the mean for the item
        ratings_wide = ratings_wide.fillna(ratings_wide.mean(axis=1))
        
        # Calculate ICCs using the formula for ICC(2,1) - two-way random effects, single rater
        # Between items variance
        ms_items = n_observers * ratings_wide.var(axis=1).sum() / (n_items - 1)
        
        # Between observers variance
        ms_observers = n_items * ratings_wide.var(axis=0).sum() / (n_observers - 1)
        
        # Residual variance
        total_sum_sq = ((ratings_wide - ratings_wide.mean().mean())**2).sum().sum()
        items_sum_sq = n_observers * ((ratings_wide.mean(axis=1) - ratings_wide.mean().mean())**2).sum()
        observers_sum_sq = n_items * ((ratings_wide.mean(axis=0) - ratings_wide.mean().mean())**2).sum()
        residual_sum_sq = total_sum_sq - items_sum_sq - observers_sum_sq
        df_residual = (n_items - 1) * (n_observers - 1)
        ms_residual = residual_sum_sq / df_residual
        
        # ICC calculations
        icc1 = (ms_items - ms_residual) / (ms_items + (n_observers - 1) * ms_residual)
        icc2 = (ms_items - ms_residual) / (ms_items + (ms_observers - ms_residual) / n_items + (n_observers - 1) * ms_residual)
        icc3 = (ms_items - ms_residual) / (ms_items + (n_observers - 1) * ms_residual)
        
        # Average measures ICCs
        icc1_k = (ms_items - ms_residual) / ms_items
        icc2_k = (ms_items - ms_residual) / (ms_items + (ms_observers - ms_residual) / n_items)
        icc3_k = (ms_items - ms_residual) / ms_items
        
        return {
            "icc1": icc1,  # One-way random effects
            "icc2": icc2,  # Two-way random effects
            "icc3": icc3,  # Two-way mixed effects
            "icc1_k": icc1_k,  # Average measures, one-way random
            "icc2_k": icc2_k,  # Average measures, two-way random
            "icc3_k": icc3_k,  # Average measures, two-way mixed
            "ms_items": ms_items,
            "ms_observers": ms_observers,
            "ms_residual": ms_residual,
            "n_items": n_items,
            "n_observers": n_observers
        }
    
    def name(self):
        return "Reliability"

# Register metrics
MetricRegistry.register(VarianceExplainedMetric)
MetricRegistry.register(EffectSizeAggregateMetric)
MetricRegistry.register(ReliabilityMetric)