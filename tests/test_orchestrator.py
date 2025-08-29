
import pytest
import yaml
from pathlib import Path
from src import orchestrator

def make_yaml(tmp_path, content):
    yaml_path = tmp_path / "test_config.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f)
    return yaml_path

def test_top_level_steps(tmp_path):
    config = {
        "steps": [
            {"component": "inference", "config": {"input_csv": "missing.csv", "output_csv": "out.csv"}}
        ]
    }
    yaml_path = make_yaml(tmp_path, config)
    result = orchestrator.orchestrate(yaml_path)
    assert isinstance(result, dict)
    assert not result["success"]
    assert any("input_csv" in e or "not found" in e for e in result["errors"])

def test_study_workflow_steps(tmp_path):
    config = {
        "study": {
            "name": "Test Study",
            "workflow": {
                "steps": [
                    {"component": "judge", "config": {"input_csv": "missing.csv", "output_csv": "out.csv", "template_dir": "templates/"}}
                ]
            }
        }
    }
    yaml_path = make_yaml(tmp_path, config)
    result = orchestrator.orchestrate(yaml_path)
    assert isinstance(result, dict)
    assert not result["success"]
    assert any("input_csv" in e or "not found" in e for e in result["errors"])

def test_missing_component(tmp_path):
    config = {"steps": [{"config": {"input_csv": "a.csv", "output_csv": "b.csv"}}]}
    yaml_path = make_yaml(tmp_path, config)
    result = orchestrator.orchestrate(yaml_path)
    assert not result["success"]
    assert any("missing component" in e or "missing component or config" in e for e in result["errors"])

def test_missing_config(tmp_path):
    config = {"steps": [{"component": "inference"}]}
    yaml_path = make_yaml(tmp_path, config)
    result = orchestrator.orchestrate(yaml_path)
    assert not result["success"]
    assert any("missing component or config" in e for e in result["errors"])

def test_unknown_component(tmp_path):
    config = {"steps": [{"component": "not_a_real_component", "config": {}}]}
    yaml_path = make_yaml(tmp_path, config)
    result = orchestrator.orchestrate(yaml_path)
    assert not result["success"]
    assert any("unknown component" in e for e in result["errors"])

def test_malformed_yaml(tmp_path):
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("steps: [ { component: inference, config: [bad] } ]: bad")
    result = orchestrator.orchestrate(yaml_path)
    assert not result["success"]
    assert any("could not" in e or "error" in e or "No steps found" in e for e in result.get("errors", []))

def test_registry_param_schema():
    reg = orchestrator.ACTIVITY_REGISTRY
    assert "inference" in reg and "param_schema" in reg["inference"]
    assert "input_csv" in reg["inference"]["param_schema"]
    assert "judge" in reg and "param_schema" in reg["judge"]
    assert "template_dir" in reg["judge"]["param_schema"]

def test_missing_required_param(tmp_path):
    config = {"steps": [{"component": "inference", "config": {"output_csv": "out.csv"}}]}
    yaml_path = make_yaml(tmp_path, config)
    result = orchestrator.orchestrate(yaml_path)
    assert not result["success"]
    assert any("missing config key" in e or "input_csv" in e for e in result["errors"])

def test_success_path_with_mock(tmp_path, monkeypatch):
    # Save the original handler
    original_handler = orchestrator.ACTIVITY_REGISTRY["inference"]["handler"]
    try:
        def mock_handler(cfg):
            return {"success": True, "artifacts": [cfg["output_csv"]]}
        orchestrator.ACTIVITY_REGISTRY["inference"]["handler"] = mock_handler
        config = {"steps": [{"component": "inference", "config": {"input_csv": "a.csv", "output_csv": "b.csv"}}]}
        yaml_path = make_yaml(tmp_path, config)
        result = orchestrator.orchestrate(yaml_path)
        assert result["success"]
        assert "b.csv" in result["artifacts"]
    finally:
        orchestrator.ACTIVITY_REGISTRY["inference"]["handler"] = original_handler

def test_empty_steps(tmp_path):
    config = {"steps": []}
    yaml_path = make_yaml(tmp_path, config)
    result = orchestrator.orchestrate(yaml_path)
    assert not result["success"]
    assert any("No steps found" in e for e in result["errors"])

def test_no_steps_found(tmp_path):
    config = {"foo": "bar"}
    yaml_path = make_yaml(tmp_path, config)
    result = orchestrator.orchestrate(yaml_path)
    assert not result["success"]
    assert any("No steps found" in e for e in result["errors"])

def test_multiple_errors(tmp_path):
    config = {"steps": [
        {"config": {}},  # missing component
        {"component": "not_a_real_component", "config": {}},
        {"component": "inference"}  # missing config
    ]}
    yaml_path = make_yaml(tmp_path, config)
    result = orchestrator.orchestrate(yaml_path)
    assert not result["success"]
    assert len(result["errors"]) >= 1

def test_register_new_activity(tmp_path):
    called = {}
    def dummy_handler(cfg):
        called["yes"] = True
        return {"success": True, "artifacts": ["dummy"]}
    orchestrator.register_activity("dummy", param_schema={"foo": {"type": "str", "required": True}})(dummy_handler)
    config = {"steps": [{"component": "dummy", "config": {"foo": "bar"}}]}
    yaml_path = make_yaml(tmp_path, config)
    result = orchestrator.orchestrate(yaml_path)
    assert result["success"]
    assert called.get("yes")
import pytest
import tempfile
import yaml
from pathlib import Path
from src import orchestrator

def make_yaml(tmp_path, content):
    yaml_path = tmp_path / "test_config.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f)
    return yaml_path

def test_orchestrator_top_level_steps(tmp_path):
    # Minimal valid config with top-level steps
    config = {
        "steps": [
            {"component": "inference", "config": {"input_csv": "dummy.csv", "output_csv": "dummy_out.csv"}}
        ]
    }
    yaml_path = make_yaml(tmp_path, config)
    result = orchestrator.orchestrate(yaml_path)
    assert isinstance(result, dict)
    assert "success" in result
    assert "errors" in result
    # Should fail due to missing input_csv, but error should be clear
    assert not result["success"]
    assert any("input_csv" in e or "not found" in e for e in result["errors"])
