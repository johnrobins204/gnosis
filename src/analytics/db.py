from abc import ABC, abstractmethod
import pandas as pd

class AnalyticsDataAccess(ABC):
    """
    Abstract interface for analytics data access (file or DB backend).
    """
    @abstractmethod
    def load_data(self, **kwargs) -> pd.DataFrame:
        """Load analytics data as a DataFrame."""
        pass

    @abstractmethod
    def save_data(self, df: pd.DataFrame, **kwargs):
        """Save analytics data from a DataFrame."""
        pass

    @abstractmethod
    def query(self, **kwargs) -> pd.DataFrame:
        """Query analytics data with filters/params."""
        pass

class FileAnalyticsDataAccess(AnalyticsDataAccess):
    """
    File-based implementation (CSV/Parquet).
    """
    def __init__(self, path, filetype="csv"):
        self.path = path
        self.filetype = filetype

    def load_data(self, **kwargs):
        if self.filetype == "csv":
            return pd.read_csv(self.path)
        elif self.filetype == "parquet":
            return pd.read_parquet(self.path)
        else:
            raise ValueError(f"Unsupported filetype: {self.filetype}")

    def save_data(self, df, **kwargs):
        if self.filetype == "csv":
            df.to_csv(self.path, index=False)
        elif self.filetype == "parquet":
            df.to_parquet(self.path, index=False)
        else:
            raise ValueError(f"Unsupported filetype: {self.filetype}")

    def query(self, **kwargs):
        df = self.load_data()
        # Simple filter: pass column=value pairs in kwargs
        for k, v in kwargs.items():
            if k in df.columns:
                df = df[df[k] == v]
        return df
