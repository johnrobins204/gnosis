from analytics.db import FileAnalyticsDataAccess
from analytics.db_postgres import PostgresAnalyticsDataAccess

def get_analytics_data_access(config: dict):
    """
    Returns the correct AnalyticsDataAccess backend based on config.
    Expects config to have 'output_format' (e.g., 'csv', 'parquet', 'json', 'postgres')
    and 'output_csv' or 'path' for file-based backends.
    """
    output_format = config.get("output_format", "csv")
    if output_format == "postgres":
        return PostgresAnalyticsDataAccess()
    else:
        # Default to file-based
        path = config.get("output_csv") or config.get("path")
        if not path:
            raise ValueError("File path must be specified for file-based analytics backend.")
        return FileAnalyticsDataAccess(path, filetype=output_format)