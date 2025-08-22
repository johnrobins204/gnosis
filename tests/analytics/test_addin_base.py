import pytest
from src.analytics.analytics_addin_base import AnalyticsAddinBase

class DummyAddin(AnalyticsAddinBase):
    def run(self, data, **kwargs):
        return "ok"

def test_dummy_addin_loads():
    addin = DummyAddin()
    assert addin.run(None) == "ok"
