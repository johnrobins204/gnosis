import os
import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from src.analytics.db_postgres import PostgresAnalyticsDataAccess

@pytest.fixture(scope="module")
def pg_test_table():
    # Use environment variables for connection
    user = os.environ.get("PGUSER", "postgres")
    password = os.environ.get("PGPASSWORD", "")
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    dbname = os.environ.get("PGDATABASE", "analytics_test")
    table = "test_metrics"
    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")
    # Create table
    with engine.connect() as conn:
        with engine.begin() as conn:  # begin() ensures commit
            conn.execute(text(f'DROP TABLE IF EXISTS {table}'))
            conn.execute(text(f'CREATE TABLE {table} (metric TEXT, value INT)'))
            conn.execute(text(f"INSERT INTO {table} (metric, value) VALUES ('A', 1), ('B', 2), ('A', 3)"))
    yield dict(user=user, password=password, host=host, port=port, dbname=dbname, table=table)
    # Teardown
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table}"))

def test_postgres_load_and_query(pg_test_table):
    dal = PostgresAnalyticsDataAccess(**pg_test_table)
    df = dal.load_data()
    assert set(df["metric"]) == {"A", "B"}
    filtered = dal.query(metric="A")
    assert len(filtered) == 2
    assert all(filtered["metric"] == "A")

def test_postgres_save(pg_test_table):
    dal = PostgresAnalyticsDataAccess(**pg_test_table)
    df = pd.DataFrame({"metric": ["X"], "value": [42]})
    dal.save_data(df, if_exists="append")
    loaded = dal.query(metric="X")
    assert len(loaded) == 1
    assert loaded.iloc[0]["value"] == 42
