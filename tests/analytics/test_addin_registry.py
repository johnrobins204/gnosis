import os
import tempfile
import pytest
from src.analytics.addin_registry import AddinRegistry
from src.analytics.analytics_addin_base import AnalyticsAddinBase

class ValidAddin(AnalyticsAddinBase):
    def run(self, data, **kwargs):
        return "ok"

class NotAnAddin:
    pass

def make_manifest(path, entry_point, name="ValidAddin"):  # Helper
    manifest = f"""
name: "{name}"
version: "0.1.0"
author: "Test <test@example.com>"
description: "Test add-in."
input_data_types:
  - DataFrame
metrics:
  - name: "test_metric"
    description: "Test metric."
dependencies:
  - pandas>=1.3.0
entry_point: "{entry_point}"
"""
    with open(path, 'w') as f:
        f.write(manifest)

def test_registry_register_and_get():
    reg = AddinRegistry()
    reg.register("valid", ValidAddin)
    assert reg.get("valid") is ValidAddin
    with pytest.raises(TypeError):
        reg.register("bad", NotAnAddin)

def test_discover_from_manifests(tmp_path, monkeypatch):
    # Write a valid manifest
    manifest_path = tmp_path / "valid.yaml"
    # Patch sys.modules so importlib can find our dummy class
    import sys
    sys.modules["dummy_module"] = type(sys)("dummy_module")
    setattr(sys.modules["dummy_module"], "ValidAddin", ValidAddin)
    make_manifest(manifest_path, "dummy_module.ValidAddin")
    reg = AddinRegistry()
    reg.discover_from_manifests(str(tmp_path))
    assert "ValidAddin" in reg.list_addins()
    # Write an invalid manifest (bad entry point)
    bad_manifest_path = tmp_path / "bad.yaml"
    make_manifest(bad_manifest_path, "dummy_module.DoesNotExist", name="BadAddin")
    reg2 = AddinRegistry()
    reg2.discover_from_manifests(str(tmp_path))
    assert "BadAddin" not in reg2.list_addins()
