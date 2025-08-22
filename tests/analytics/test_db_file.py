import os
import pandas as pd
import pytest
from src.analytics.db import FileAnalyticsDataAccess

def test_fileanalytics_load_and_query(tmp_path):
    # Create a sample CSV
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        "Metric": ["A", "B", "A"],
        "Value": [1, 2, 3]
    })
    df.to_csv(csv_path, index=False)
    dal = FileAnalyticsDataAccess(str(csv_path), filetype="csv")
    loaded = dal.load_data()
    assert loaded.equals(df)
    # Query for Metric == 'A'
    filtered = dal.query(Metric="A")
    assert len(filtered) == 2
    assert all(filtered["Metric"] == "A")

def test_fileanalytics_save(tmp_path):
    csv_path = tmp_path / "out.csv"
    dal = FileAnalyticsDataAccess(str(csv_path), filetype="csv")
    df = pd.DataFrame({"Metric": ["X"], "Value": [42]})
    dal.save_data(df)
    loaded = pd.read_csv(csv_path)
    assert loaded.equals(df)
