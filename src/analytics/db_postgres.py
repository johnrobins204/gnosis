
import os
import pandas as pd  # type: ignore
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db_migrations.models import AnalyticsEvent, Experiment, ExperimentResult, AggregationResult
from .db import AnalyticsDataAccess

class PostgresAnalyticsDataAccess(AnalyticsDataAccess):
    """
    PostgreSQL implementation of AnalyticsDataAccess using SQLAlchemy ORM.
    """
    def __init__(self):
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        db = os.getenv("DB_NAME")
        self.conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
        self.engine = create_engine(self.conn_str)
        self.Session = sessionmaker(bind=self.engine)

    def load_data(self, **kwargs):
        session = self.Session()
        try:
            query = session.query(AnalyticsEvent)
            for k, v in kwargs.items():
                query = query.filter(getattr(AnalyticsEvent, k) == v)
            results = query.all()
            df = pd.DataFrame([r.__dict__ for r in results])
            if "_sa_instance_state" in df.columns:
                df = df.drop(columns=["_sa_instance_state"])
            return df
        finally:
            session.close()

    def save_data(self, df: pd.DataFrame, **kwargs):
        session = self.Session()
        try:
            for _, row in df.iterrows():
                event = AnalyticsEvent(**row.to_dict())
                session.add(event)
            session.commit()
        finally:
            session.close()

    def query(self, **kwargs):
        return self.load_data(**kwargs)

    # Experiment tracking
    def save_experiment(self, fingerprint, config_json):
        session = self.Session()
        try:
            exp = Experiment(fingerprint=fingerprint, config_json=config_json)
            session.add(exp)
            session.commit()
            return exp.id
        finally:
            session.close()

    def save_experiment_result(self, experiment_id, results_json, tags=None):
        session = self.Session()
        try:
            res = ExperimentResult(experiment_id=experiment_id, results_json=results_json, tags=tags)
            session.add(res)
            session.commit()
            return res.id
        finally:
            session.close()

    def save_aggregation_result(self, experiment_id, stage_name, result_json):
        session = self.Session()
        try:
            agg = AggregationResult(experiment_id=experiment_id, stage_name=stage_name, result_json=result_json)
            session.add(agg)
            session.commit()
            return agg.id
        finally:
            session.close()
