import pytest
from src.analytics.metrics import semantic

def test_semantic_metric_import():
    # Just ensure the module imports and exposes expected attributes
    assert hasattr(semantic, "__doc__")
