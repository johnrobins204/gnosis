import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from .db import AnalyticsDataAccess

class PostgresAnalyticsDataAccess(AnalyticsDataAccess):
    """
    PostgreSQL implementation of AnalyticsDataAccess using SQLAlchemy.
    """
    def __init__(self, user, password, host, port, dbname, table):
        self.conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        self.engine: Engine = create_engine(self.conn_str)
        self.table = table

    def load_data(self, **kwargs):
        # Load entire table as DataFrame
        return pd.read_sql_table(self.table, self.engine)

    def save_data(self, df: pd.DataFrame, if_exists="append", **kwargs):
        df.to_sql(self.table, self.engine, if_exists=if_exists, index=False)

    def query(self, **kwargs):
        # Simple filter: pass column=value pairs in kwargs
        query = f"SELECT * FROM {self.table}"
        if kwargs:
            filters = [f"{k} = '{v}'" for k, v in kwargs.items()]
            query += " WHERE " + " AND ".join(filters)
        return pd.read_sql_query(query, self.engine)
