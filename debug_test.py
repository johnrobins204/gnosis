import yaml
from pathlib import Path
from src import orchestrator

def make_yaml(tmp_path, content):
    yaml_path = tmp_path / "test_config.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f)
    return yaml_path

# Test missing component
tmp_path = Path("/tmp")
config = {"steps": [{"config": {"input_csv": "a.csv", "output_csv": "b.csv"}}]}
yaml_path = make_yaml(tmp_path, config)
result = orchestrator.orchestrate(yaml_path)
print("Result:", result)
print("Success:", result["success"])
print("Errors:", result["errors"])
