# Database Migration and Analytics I/O Usage Guide

This document describes the database migration process and the updated usage patterns for analytics data, experiment tracking, and aggregation results in gnosisPWB.

## Database Migration Process

### 1. Alembic Migration System
- Alembic is used for managing schema migrations for the analytics database.
- Migration scripts are located in `db_migrations/migrations/versions/`.
- The SQLAlchemy models are defined in `db_migrations/models.py`.

### 2. Running Migrations
- Ensure your database connection details are set via environment variables:
  - `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`
- To apply all migrations:
  ```bash
  alembic -c db_migrations/alembic.ini upgrade head
  ```
- To create a new migration after model changes:
  ```bash
  alembic -c db_migrations/alembic.ini revision --autogenerate -m "Describe your change"
  alembic -c db_migrations/alembic.ini upgrade head
  ```

## Analytics I/O Usage

### 1. Analytics Data Access
- Use `PostgresAnalyticsDataAccess` (in `src/analytics/db_postgres.py`) for all analytics event I/O.
- Example:
  ```python
  from src.analytics.db_postgres import PostgresAnalyticsDataAccess
  db = PostgresAnalyticsDataAccess()
  df = db.load_data(event_type="inference")
  db.save_data(df)
  ```

### 2. Experiment Tracking
- Use `ExperimentTracker` (in `src/analytics/experiment_tracker.py`) for experiment and result tracking.
- By default, uses the database backend if available.
- Example:
  ```python
  from src.analytics.experiment_tracker import ExperimentTracker
  tracker = ExperimentTracker(use_db=True)
  fingerprint = tracker.generate_fingerprint(config)
  tracker.save_results(fingerprint, results, tags=["baseline"], config=config)
  loaded = tracker.load_results(fingerprint)
  ```

### 3. Aggregation Results
- Use the `DataAggregator` class (in `src/analytics/aggregation.py`) to run analytics pipelines and optionally write aggregation results to the database.
- Example:
  ```python
  from src.analytics.aggregation import DataAggregator
  aggregator = DataAggregator()
  # ... run pipeline ...
  aggregator.save_aggregation_result_to_db(experiment_id, stage_name, result_df)
  ```

## Environment Variables
- Set the following environment variables for DB access:
  - `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`
- Use a `.env` file or export variables in your shell.

## Troubleshooting
- If migrations fail, check DB credentials and Alembic config.
- Ensure all dependencies are installed (`pip install -r requirements.txt`).
- For further help, see `db_migrations/README.md`.

---
For more details, see the code and docstrings in the relevant modules.
