
import hashlib
import json
import os
from typing import Optional
from .db_postgres import PostgresAnalyticsDataAccess
from db_migrations.models import Experiment, ExperimentResult

class ExperimentTracker:
    """Class for tracking experiments and their results (DB or file backend)."""

    def __init__(self, storage_path: str = "experiments/", use_db: bool = True) -> None:
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
        self.use_db = use_db
        self.db = PostgresAnalyticsDataAccess() if use_db else None

    def generate_fingerprint(self, config: dict) -> str:
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def save_results(self, fingerprint: str, results: dict, tags: Optional[list] = None, config: Optional[dict] = None) -> None:
        if self.use_db and self.db:
            # Save experiment and result to DB
            config_json = json.dumps(config) if config else "{}"
            exp_id = self.db.save_experiment(fingerprint, config_json)
            self.db.save_experiment_result(exp_id, json.dumps(results), tags=','.join(tags) if tags else None)
        else:
            file_path = os.path.join(self.storage_path, f"{fingerprint}.json")
            experiment_data = {"results": results, "tags": tags or []}
            with open(file_path, "w") as file:
                json.dump(experiment_data, file, indent=4)

    def load_results(self, fingerprint: str) -> dict:
        if self.use_db and self.db:
            # Load from DB
            session = self.db.Session()
            try:
                exp = session.query(Experiment).filter_by(fingerprint=fingerprint).first()
                if not exp:
                    raise FileNotFoundError(f"No results found for fingerprint: {fingerprint}")
                res = session.query(ExperimentResult).filter_by(experiment_id=exp.id).order_by(ExperimentResult.created_at.desc()).first()
                if not res:
                    raise FileNotFoundError(f"No results found for fingerprint: {fingerprint}")
                return json.loads(str(res.results_json))
            finally:
                session.close()
        else:
            file_path = os.path.join(self.storage_path, f"{fingerprint}.json")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"No results found for fingerprint: {fingerprint}")
            with open(file_path, "r") as file:
                return json.load(file)

    def list_experiments(self) -> list:
        if self.use_db and self.db:
            session = self.db.Session()
            try:
                exps = session.query(Experiment).all()
                return [e.fingerprint for e in exps]
            finally:
                session.close()
        else:
            return [f.split(".json")[0] for f in os.listdir(self.storage_path) if f.endswith(".json")]

    def filter_experiments(self, tag: str) -> list:
        if self.use_db and self.db:
            session = self.db.Session()
            try:
                res = session.query(ExperimentResult).filter(ExperimentResult.tags.like(f"%{tag}%")).all()
                exp_ids = set(r.experiment_id for r in res)
                exps = session.query(Experiment).filter(Experiment.id.in_(exp_ids)).all()
                return [e.fingerprint for e in exps]
            finally:
                session.close()
        else:
            experiments = []
            for file_name in os.listdir(self.storage_path):
                if file_name.endswith(".json"):
                    with open(os.path.join(self.storage_path, file_name), "r") as file:
                        data = json.load(file)
                        if tag in data.get("tags", []):
                            experiments.append(file_name.split(".json")[0])
            return experiments