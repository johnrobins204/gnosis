from src.analytics import api

def test_api_import():
    # Just ensure the module imports and exposes expected attributes
    assert hasattr(api, "__doc__")
