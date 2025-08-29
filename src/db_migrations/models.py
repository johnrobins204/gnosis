from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class AnalyticsEvent(Base):
    __tablename__ = 'analytics_events'
    id = Column(Integer, primary_key=True)
    event_type = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    value = Column(Float)

class Experiment(Base):
    __tablename__ = 'experiments'
    id = Column(Integer, primary_key=True)
    fingerprint = Column(String, unique=True, nullable=False)
    config_json = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class ExperimentResult(Base):
    __tablename__ = 'experiment_results'
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, nullable=False)
    results_json = Column(String, nullable=False)
    tags = Column(String)  # Comma-separated tags for simplicity
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class AggregationResult(Base):
    __tablename__ = 'aggregation_results'
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, nullable=True)
    stage_name = Column(String, nullable=False)
    result_json = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
