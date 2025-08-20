import yaml
import os

class AnalyticsConfig:
    DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "default_analytics_config.yaml")

    def __init__(self, config_path=None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = self._load_config()
        self.validate_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)

    def get_metric_status(self, category):
        """Check if a metric category is enabled."""
        return self.config.get("metrics", {}).get(category, {}).get("enabled", False)

    def validate_config(self):
        """Validate the configuration for any inconsistencies."""
        if "metrics" not in self.config:
            raise ValueError("Configuration must include a 'metrics' section.")
        for category, settings in self.config["metrics"].items():
            if not isinstance(settings, dict) or "enabled" not in settings:
                raise ValueError(f"Invalid configuration for metric category: {category}")

# Example usage:
# config = AnalyticsConfig()
# print(config.get_metric_status("nlp"))