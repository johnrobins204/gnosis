# Database Migration, Audit, and Test Plan

This document consolidates all migration, audit, and test planning for the transition from file-based to database-backed analytics, experiment, and aggregation data in the Gnosis project.

---

## 1. File-based I/O Locations
- `src/io.py`: `load_csv` (pandas.read_csv), `write_dataframe` (pandas.DataFrame.to_csv), open/json.dump for meta
- `src/analytics/experiment_tracker.py`: open/json.dump/json.load for experiment results (experiments/{fingerprint}.json)
- `src/analytics/db.py`: FileAnalyticsDataAccess uses pandas.read_csv, pandas.read_parquet, DataFrame.to_csv, DataFrame.to_parquet
- `src/analytics/aggregation.py`: likely stores aggregation results as DataFrames (CSV/Parquet/JSON)
- `src/judge.py`: uses load_csv, write_dataframe for input/output CSVs
- `src/analytics/yaml_loader.py`: open for YAML config
- General: open/json.dump/json.load in various modules for config, manifest, and meta file handling

---

## 2. Database-backed I/O Locations
- `src/analytics/experiment_tracker.py`: PostgresAnalyticsDataAccess, SQLAlchemy Session/query for experiment/results (Experiment, ExperimentResult tables)
- `src/analytics/db_postgres.py`: AnalyticsDataAccess for analytics data (AnalyticsEvent, AggregationResult, etc.), save_experiment, save_experiment_result, save_aggregation_result
- `src/analytics/aggregation.py`: DataAggregator.save_aggregation_result_to_db writes aggregation results to DB (AggregationResult table)
- `src/models/impl.py`: get_experiment_model_instance can use experiment_adapter for DB-backed experiment models
- General: SQLAlchemy ORM usage for all DB-backed analytics, experiment, and aggregation data

---

## 3. File vs DB Feature/Field Comparison
### Experiment Tracking
- File-based: experiments/{fingerprint}.json
  - Fields: results (dict), tags (list), config (dict, optional)
  - One result per file (no versioning)
- DB-backed: Experiment, ExperimentResult tables
  - Fields: fingerprint (str), config_json (str/JSON), results_json (str/JSON), tags (str, comma-separated), created_at (timestamp)
  - Supports multiple results per experiment (versioning/history)

### Analytics Data
- File-based: CSV/Parquet via pandas
  - Fields: columns as in DataFrame, flexible schema
  - Simple filter/query via pandas
- DB-backed: AnalyticsEvent table (ORM)
  - Fields: columns as in ORM model, fixed schema
  - Complex queries/joins supported

### Aggregation Results
- File-based: DataFrames (CSV/Parquet/JSON)
  - Fields: columns as in DataFrame, flexible schema
- DB-backed: AggregationResult table
  - Fields: experiment_id, stage_name, result_json (JSON), created_at
  - Results stored as JSON blobs, with experiment/stage metadata

---

## 4. Gaps and Differences
- DB supports multiple results per experiment (versioning/history); file-based supports only one result per file.
- File-based tags are a list; DB stores tags as a comma-separated string (requires conversion during migration).
- DB records created_at timestamps for results; file-based does not.
- DB can enforce schema and relationships; file-based is schema-less.
- File-based schema is flexible (DataFrame columns); DB schema is fixed (ORM model).
- DB supports complex queries and joins; file-based supports only simple filters.
- Migration must ensure all columns in CSV/Parquet are mapped to DB fields.
- File-based may use CSV/Parquet/JSON; DB stores results as JSON blobs with experiment/stage metadata.
- DB supports linking aggregation results to experiments and stages; file-based may not have this metadata.
- Migration scripts must handle data type conversions (e.g., lists to strings, timestamps, JSON encoding).
- Some features (e.g., versioning, metadata, relationships) are only available in DB and may require new logic in the application to fully leverage.

---

## 5. Migration Script Plan
- Script will scan `experiments/` directory for all `*.json` files and migrate to DB.
- Script will scan for all analytics CSV/Parquet files and migrate to DB.
- Script will scan for all aggregation result files (CSV/Parquet/JSON) and migrate to DB.
- Scripts will log all actions, support dry-run mode, and optionally backup files before migration.

---

## 6. Migration and I/O Usage Documentation
- All analytics, experiment, and aggregation data should be read from and written to the PostgreSQL database using the provided data access classes (`PostgresAnalyticsDataAccess`, `ExperimentTracker`, etc.).
- File-based storage (CSV, Parquet, JSON) is deprecated for production use but may be supported for import/export and legacy compatibility.
- All new data should be written to the database, and all queries should use the ORM/data access layer.
- See migration process steps in this document for details.

---

## 7. Test Plan
- Ensure all major I/O operations are tested for both file-based and DB-backed backends.
- Validate correctness of migration scripts and data integrity post-migration.
- Use pytest parameterization to run tests for both backends.
- Add/expand tests in `tests/analytics/`, `tests/inference/`, and `tests/judge/` as needed.
- Document any backend-specific test logic in test files and README.

---

## Audit Trail
- All steps, findings, and plans are versioned and traceable in the `docs/` folder.
- This ensures full auditability and reproducibility for T2P and future maintenance.
