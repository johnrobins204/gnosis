from src.analytics import config

def test_config_import():
    # Just ensure the module imports and exposes expected attributes
    assert hasattr(config, "__doc__")
