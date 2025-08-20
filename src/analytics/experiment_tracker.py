import hashlib
import json
import os

class ExperimentTracker:
    """Class for tracking experiments and their results."""

    def __init__(self, storage_path="experiments/"):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

    def generate_fingerprint(self, config):
        """Generate a unique fingerprint for an experiment configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def save_results(self, fingerprint, results, tags=None):
        """Save experiment results with optional tags."""
        file_path = os.path.join(self.storage_path, f"{fingerprint}.json")
        experiment_data = {"results": results, "tags": tags or []}
        with open(file_path, "w") as file:
            json.dump(experiment_data, file, indent=4)

    def load_results(self, fingerprint):
        """Load experiment results from a file."""
        file_path = os.path.join(self.storage_path, f"{fingerprint}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No results found for fingerprint: {fingerprint}")
        with open(file_path, "r") as file:
            return json.load(file)

    def list_experiments(self):
        """List all tracked experiments."""
        return [f.split(".json")[0] for f in os.listdir(self.storage_path) if f.endswith(".json")]

    def filter_experiments(self, tag):
        """Filter experiments by a specific tag."""
        experiments = []
        for file_name in os.listdir(self.storage_path):
            if file_name.endswith(".json"):
                with open(os.path.join(self.storage_path, file_name), "r") as file:
                    data = json.load(file)
                    if tag in data.get("tags", []):
                        experiments.append(file_name.split(".json")[0])
        return experiments