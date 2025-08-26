from .db_postgres import PostgresAnalyticsDataAccess
import json
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Callable, Union, Any, Optional

from typing import List, Dict, Callable, Union, Any, Optional
import pandas as pd
import numpy as np

class DataAggregator:
    def save_aggregation_result_to_db(self, experiment_id, stage_name, result_df):
        db = PostgresAnalyticsDataAccess()
        result_json = result_df.to_json(orient="records")
        db.save_aggregation_result(experiment_id, stage_name, result_json)
    """
    System for flexible multi-level data aggregation in experimental analytics.
    Supports building analytics pipelines with operations at different levels of granularity.
    """

    aggregation_functions: Dict[str, Callable[[Any], Any]]

    def __init__(self) -> None:
        self.aggregation_functions: Dict[str, Callable[[Any], Any]] = {
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

    def register_aggregation_function(self, name: str, func: Callable[[Any], Any]) -> None:
        """Register a custom aggregation function."""
        self.aggregation_functions[name] = func

    def multi_level_pipeline(
        self,
        data: pd.DataFrame,
        pipeline_config: List[Dict[str, Any]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute a multi-level analytics pipeline, performing operations at different levels of aggregation.
        """
        results: Dict[str, pd.DataFrame] = {}
        current_data: pd.DataFrame = data.copy()

        for stage in pipeline_config:
            stage_name: str = stage["name"]
            stage_data: pd.DataFrame = current_data.copy()

            # Apply filter if specified
            if "filter" in stage and stage["filter"]:
                filter_func: Callable[[pd.Series], bool] = stage["filter"]
                stage_data = stage_data[stage_data.apply(filter_func, axis=1)]

            # Apply transform if specified
            if "transform" in stage and stage["transform"]:
                transform_func: Callable[[pd.DataFrame], pd.DataFrame] = stage["transform"]
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

            # Optionally write to DB if requested
            if stage.get("write_to_db"):
                experiment_id = stage.get("experiment_id")
                self.save_aggregation_result_to_db(experiment_id, stage_name, stage_data)

            # If this stage is marked as input for next stage, update current_data
            if stage.get("output_to_next", False):
                current_data = stage_data

        return results

    def save_pipeline_config(self, config: List[Dict[str, Any]], filepath: str) -> None:
        """Save a pipeline configuration to a JSON file for reproducibility."""
        import json

        def serialize_config(config_item: Any) -> Any:
            if isinstance(config_item, dict):
                return {k: serialize_config(v) for k, v in config_item.items()}
            elif isinstance(config_item, list):
                return [serialize_config(item) for item in config_item]
            elif callable(config_item) and not isinstance(config_item, type):
                return f"callable:{getattr(config_item, '__name__', str(config_item))}"
            else:
                return config_item

        serialized_config = serialize_config(config)

        with open(filepath, 'w') as f:
            json.dump(serialized_config, f, indent=2)

    def aggregate(
        self,
        data: pd.DataFrame,
        group_by: Union[str, List[str]],
        metrics: Dict[str, Union[str, Callable[[pd.Series], Any]]]
    ) -> pd.DataFrame:
        """
        Aggregate data by group_by columns using specified metrics.
        """
        if isinstance(group_by, str):
            group_by = [group_by]
        if not isinstance(metrics, dict):
            raise ValueError("metrics must be a dict of {output_col: agg_func_or_str}")
        if data.empty:
            return pd.DataFrame(columns=group_by + list(metrics.keys()))

        agg_dict: Dict[str, Union[tuple, Callable[[pd.Series], Any]]] = {}
        for out_col, agg in metrics.items():
            if isinstance(agg, str):
                if agg in self.aggregation_functions:
                    func = self.aggregation_functions[agg]
                    col_candidates = [out_col]
                    if '_' in out_col:
                        col_candidates.append(out_col.split('_')[-1])
                    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
                    col_candidates += numeric_cols
                    found_col: Optional[str] = None
                    for c in col_candidates:
                        if c in data.columns and c in numeric_cols:
                            found_col = c
                            break
                    if not found_col:
                        raise KeyError(f"No numeric column found for metric '{out_col}' in data columns {list(data.columns)}")
                    agg_dict[out_col] = (found_col, func)
                else:
                    raise KeyError(f"Unknown aggregation function: {agg}")
            elif callable(agg):
                agg_dict[out_col] = agg
            else:
                raise ValueError(f"Metric value for '{out_col}' must be str or callable")

        named_aggs = {k: v for k, v in agg_dict.items() if isinstance(v, tuple)}
        custom_aggs = {k: v for k, v in agg_dict.items() if callable(v)}

        result_df: pd.DataFrame
        if named_aggs:
            # mypy: pandas groupby.agg with named tuples is not statically typed
            result_df = data.groupby(group_by).agg(**named_aggs).reset_index()  # type: ignore
        else:
            result_df = data.groupby(group_by).size().reset_index().iloc[0:0]

        if custom_aggs:
            grouped = data.groupby(group_by)
            for out_col, func in custom_aggs.items():
                # mypy: pandas groupby.apply is not statically typed
                custom_result = grouped.apply(func)  # type: ignore
                if not isinstance(custom_result, pd.DataFrame):
                    custom_result = custom_result.to_frame(out_col)
                else:
                    custom_result = custom_result.rename(columns={0: out_col})
                custom_result = custom_result.reset_index()
                result_df = pd.merge(result_df, custom_result, on=group_by, how="outer")

        # Ensure result_df is always a DataFrame
        if not isinstance(result_df, pd.DataFrame):
            raise RuntimeError("Aggregation did not produce a DataFrame result.")
        return result_df