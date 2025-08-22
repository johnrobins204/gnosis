from src.analytics.config import validate_config
from src.analytics.aggregation import DataAggregator
from src.analytics.registry import MetricRegistry
from src.io import load_csv, write_dataframe

class AnalyticsAPI:
    @staticmethod
    def run_from_config(cfg: dict) -> dict:
        validate_config(cfg)
        data = load_csv(cfg["input_csv"])
        aggregator = DataAggregator()
        metrics = MetricRegistry.get_metrics(cfg.get("metrics"))
        result_df = aggregator.aggregate(data, metrics, group_by=cfg["group_col"])
        write_dataframe(result_df, cfg["output_csv"])
        return {"success": True, "artifacts": [cfg["output_csv"]], "rows": len(result_df)}

    @staticmethod
    def run_cli(argv=None):
        # argparse logic here, calls run_from_config
        pass