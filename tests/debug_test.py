import pytest
import yaml
from pathlib import Path
from src import orchestrator

def make_yaml(tmp_path, content):
    yaml_path = tmp_path / "test_config.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f)
    return yaml_path

def test_missing_component(tmp_path):
    config = {"steps": [{"config": {"input_csv": "a.csv", "output_csv": "b.csv"}}]}
    yaml_path = make_yaml(tmp_path, config)
    result = orchestrator.orchestrate(yaml_path)
    print("DEBUG RESULT:", result)
    assert not result["success"]
    assert any("missing component" in e or "missing component or config" in e for e in result["errors"])
