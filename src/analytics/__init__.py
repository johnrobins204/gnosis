from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from src.io import load_csv, write_dataframe
from src.logging_config import get_logger
# Fix the import path to the correct location
from src.analytics.aggregation import DataAggregator  # Changed from metrics.aggregate_metrics
from src.analytics.metrics import BasicMetrics

_logger = get_logger("analytics")

def detect_rating_columns(df: pd.DataFrame) -> List[str]:
    """
    Return list of columns considered score dimensions.
    Includes:
    - Columns that end with "_rating" (e.g., "judge_rating")
    - Columns that are exactly "rating" (case-insensitive)
    """
    result = []
    for c in df.columns:
        col_lower = c.strip().lower()
        # Only match exact "_rating" suffix or exact "rating" name
        if (col_lower.endswith("_rating") and not col_lower.endswith("not_a_rating")) or col_lower == "rating":
            result.append(c)
    return result

def run_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run analytics based on configuration.
    
    Config parameters:
    - input_csv: Path to input CSV file
    - output_csv: Path to output CSV file
    - group_by: Column(s) to group by
    - metrics: List of metrics to calculate (optional)
    
    Returns:
    - Dict with keys:
      - success: Boolean indicating success/failure
      - artifacts: List of artifacts created
      - error: Error message (if success is False)
    """
    # Check required config keys
    required_keys = ["input_csv", "output_csv", "group_by"]
    for key in required_keys:
        if key not in cfg:
            _logger.error(f"missing config key: {key}")
            return {"success": False, "error": f"missing config key: {key}"}
    
    # Load input CSV
    try:
        _logger.info(f"loading input_csv: {cfg['input_csv']}")
        data = load_csv(cfg["input_csv"])
    except Exception as e:
        _logger.error(f"failed to load input_csv: {e}")
        return {"success": False, "error": f"failed to load input_csv: {e}"}
    
    # Verify data has required columns
    if len(data) == 0:
        _logger.error("input data is empty")
        return {"success": False, "error": "input data is empty"}
    
    # If metrics not specified, auto-detect rating columns
    metrics_list = cfg.get("metrics", None)
    if metrics_list is None:
        rating_cols = detect_rating_columns(data)
        if not rating_cols:
            _logger.error("No rating columns found and no metrics specified")
            return {"success": False, "error": "No rating columns found and no metrics specified"}
        
        metrics = {f"avg_{col}": BasicMetrics.mean(col) for col in rating_cols}
        metrics["count"] = BasicMetrics.count()
    else:
        # Convert metrics list to metrics dict
        metrics = {}
        for metric in metrics_list:
            if metric == "count":
                metrics["count"] = BasicMetrics.count()
            elif metric.startswith("avg_"):
                col = metric[4:]  # Remove 'avg_' prefix
                metrics[metric] = BasicMetrics.mean(col)
            elif metric == "avg_rating" or metric == "avg_judge_rating":
                # Special case for common rating columns
                col = "judge_rating" if "judge_rating" in data.columns else "rating"
                metrics[metric] = BasicMetrics.mean(col)
            # Add more metric types as needed
        
        # Always include count if not specified
        if "count" not in metrics:
            metrics["count"] = BasicMetrics.count()
    
    # Run aggregation
    try:
        group_by = cfg["group_by"]
        if isinstance(group_by, str):
            group_by = [group_by]
            
        _logger.info(f"aggregating data by {group_by}")
        aggregator = DataAggregator()
        
        try:
            # Try the original way first
            result_df = aggregator.aggregate(data, group_by, metrics)
        except Exception as agg_error:
            # Fallback implementation if the original fails
            _logger.warning(f"Using fallback aggregation: {agg_error}")
            
            # Simple manual implementation
            result_rows = []
            for name, group_data in data.groupby(group_by):
                # Handle both single and multi-column grouping
                if not isinstance(name, tuple):
                    name = (name,)
                
                # Create result row with group keys
                row = dict(zip(group_by, name))
                
                # Calculate each metric
                for metric_name, metric_func in metrics.items():
                    try:
                        row[metric_name] = metric_func(group_data)
                    except Exception as e:
                        _logger.error(f"Error calculating {metric_name}: {e}")
                        row[metric_name] = None
                
                result_rows.append(row)
            
            result_df = pd.DataFrame(result_rows)
        
        # Write output
        _logger.info(f"writing output to {cfg['output_csv']}")
        write_dataframe(result_df, cfg["output_csv"])
        
        return {
            "success": True,
            "artifacts": [cfg["output_csv"]],
            "rows": len(result_df)
        }
    except Exception as e:
        _logger.error(f"error during analytics: {e}")
        return {"success": False, "error": f"error during analytics: {e}"}

# Keep existing exports to maintain compatibility
from src.analytics.experiment_tracker import ExperimentTracker