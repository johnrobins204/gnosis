from analytics.config import validate_config
from analytics.aggregation import DataAggregator
from analytics.registry import MetricRegistry
from gnosis_io import load_csv, write_dataframe

from typing import Any, Dict, Optional


class AnalyticsAPI:
    @staticmethod
    def run_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
        validate_config(cfg)
        data = load_csv(cfg["input_csv"])
        aggregator = DataAggregator()
        metrics = MetricRegistry.get_metrics(cfg.get("metrics"))
        result_df = aggregator.aggregate(data, cfg["group_col"], metrics)  # type: ignore
        write_dataframe(result_df, cfg["output_csv"])
        return {"success": True, "artifacts": [cfg["output_csv"]], "rows": len(result_df)}

    @staticmethod
    def run_cli(argv: Optional[Any] = None) -> None:
        # argparse logic here, calls run_from_config
        pass