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
    
    def aggregate(self, data: pd.DataFrame, 
                 group_by: List[str], 
                 metrics: Dict[str, Union[str, Callable]]) -> pd.DataFrame:
        """
        Aggregate data by specified columns using the given metrics.
        
        Args:
            data: DataFrame containing the data to aggregate
            group_by: List of column names to group by
            metrics: Dictionary mapping output column names to aggregation functions
                     Functions can be strings (referring to built-in functions) or callables
        
        Returns:
            DataFrame with aggregated results
        """
        # Input validation
        if data.empty:
            return pd.DataFrame(columns=group_by + list(metrics.keys()))
            
        for col in group_by:
            if col not in data.columns:
                raise ValueError(f"Group-by column '{col}' not found in data")
        
        # Create a grouped object
        grouped = data.groupby(group_by)
        
        # Create result dataframe with only the group_by columns
        result = grouped.first().reset_index()[group_by].copy()
        
        # Apply each metric
        for output_col, func_name in metrics.items():
            # Get the aggregation function
            if isinstance(func_name, str):
                if func_name not in self.aggregation_functions:
                    raise ValueError(f"Unknown aggregation function: {func_name}")
                func = self.aggregation_functions[func_name]
            else:
                func = func_name
            
            # For count, we need to count any column
            if func == len:
                result[output_col] = grouped.size().values
            else:
                # Apply function to all numeric columns and take the first one
                # This works for most metrics like mean, std, etc.
                numeric_cols = data.select_dtypes(include=np.number).columns
                
                # Make sure we have at least one numeric column
                if len(numeric_cols) == 0:
                    raise ValueError("No numeric columns found for aggregation")
                
                # Use 'score' column if it exists, otherwise use the first numeric column
                agg_col = 'score' if 'score' in numeric_cols else numeric_cols[0]
                
                # Always use the function object for custom aggregations
                # Only use string names for built-in pandas aggregations
                if isinstance(func_name, str) and func_name in ['mean', 'median', 'std', 'min', 'max', 'sum', 'var']:
                    result[output_col] = grouped[agg_col].agg(func_name).values
                else:
                    result[output_col] = grouped[agg_col].agg(func).values
    
        return result
    
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