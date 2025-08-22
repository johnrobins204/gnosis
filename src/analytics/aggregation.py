import pandas as pd
import numpy as np
from typing import List, Dict, Callable, Union, Any, Optional

class DataAggregator:
    """
    System for flexible multi-level data aggregation in experimental analytics.
    Supports building analytics pipelines with operations at different levels of granularity.
    """
    
    def __init__(self):
        self.aggregation_functions = {
            "mean": np.mean,
            "median": np.median,
            "std": np.std,
            "min": np.min,
            "max": np.max,
            "count": len,
            "sum": np.sum,
            "var": np.var,
            "sem": lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
        }
    
    def register_aggregation_function(self, name: str, func: Callable):
        """Register a custom aggregation function."""
        self.aggregation_functions[name] = func
    
    def multi_level_pipeline(self, data: pd.DataFrame, pipeline_config: List[Dict]) -> Dict[str, pd.DataFrame]:
        """
        Execute a multi-level analytics pipeline, performing operations at different levels of aggregation.
        
        Args:
            data: Initial DataFrame containing raw experimental data
            pipeline_config: List of pipeline stage configurations, each containing:
                - name: Stage name (identifier for results)
                - transform: Optional function to transform data before metrics
                - metrics: Dict of metrics to calculate at this stage
                - group_by: Optional columns to group by (if aggregation is needed)
                - filter: Optional filter to apply before processing
                - output_to_next: Whether to use this stage's output as input to the next stage
        
        Returns:
            Dictionary mapping stage names to result DataFrames
        """
        results = {}
        current_data = data.copy()
        
        for stage in pipeline_config:
            stage_name = stage["name"]
            stage_data = current_data.copy()
            
            # Apply filter if specified
            if "filter" in stage and stage["filter"]:
                filter_func = stage["filter"]
                stage_data = stage_data[stage_data.apply(filter_func, axis=1)]
            
            # Apply transform if specified
            if "transform" in stage and stage["transform"]:
                transform_func = stage["transform"]
                stage_data = transform_func(stage_data)
            
            # Apply metrics at this level
            if "metrics" in stage and stage["metrics"]:
                # If we're grouping, aggregate with metrics
                if "group_by" in stage and stage["group_by"]:
                    group_by = stage["group_by"]
                    metrics = stage["metrics"]
                    stage_data = self.aggregate(stage_data, group_by, metrics)
                # Otherwise apply metrics to each row
                else:
                    for col_name, metric_func in stage["metrics"].items():
                        if isinstance(metric_func, str):
                            if metric_func in self.aggregation_functions:
                                func = self.aggregation_functions[metric_func]
                            else:
                                raise ValueError(f"Unknown function: {metric_func}")
                        else:
                            func = metric_func
                        stage_data[col_name] = stage_data.apply(func, axis=1)
            
            # Store results for this stage
            results[stage_name] = stage_data
            
            # If this stage is marked as input for next stage, update current_data
            if stage.get("output_to_next", False):
                current_data = stage_data
            
        return results
    
    def save_pipeline_config(self, config: List[Dict], filepath: str):
        """Save a pipeline configuration to a JSON file for reproducibility."""
        import json
        
        # Convert any callable objects to their string representation
        def serialize_config(config_item):
            if isinstance(config_item, dict):
                return {k: serialize_config(v) for k, v in config_item.items()}
            elif isinstance(config_item, list):
                return [serialize_config(item) for item in config_item]
            elif callable(config_item) and not isinstance(config_item, type):
                return f"callable:{config_item.__name__}"
            else:
                return config_item
        
        serialized_config = serialize_config(config)
        
        with open(filepath, 'w') as f:
            json.dump(serialized_config, f, indent=2)
    
    def aggregate(self, data: pd.DataFrame, metrics, group_by):
        # Allow group_by to be a string or a list
        if isinstance(group_by, str):
            group_by = [group_by]
        results = []
        for metric in metrics:
            if metric in data.columns:
                grouped = data.groupby(group_by)[metric].mean().reset_index()
                grouped = grouped.rename(columns={metric: f"{metric}_mean"})
                results.append(grouped)
        # Merge all results on group_by
        if results:
            result_df = results[0]
            for df in results[1:]:
                result_df = pd.merge(result_df, df, on=group_by)
            return result_df
        else:
            return pd.DataFrame()