import logging
import pytest
from src.analytics.analytics_addin_base import AnalyticsAddinBase
from src.analytics.addin_runner import run_addin_with_logging

class GoodAddin(AnalyticsAddinBase):
    def run(self, data, **kwargs):
        return "success"

class FailingAddin(AnalyticsAddinBase):
    def run(self, data, **kwargs):
        raise ValueError("fail!")

def test_run_addin_success():
    addin = GoodAddin()
    result, error = run_addin_with_logging(addin, None)
    assert result == "success"
    assert error is None

def test_run_addin_failure(caplog):
    addin = FailingAddin()
    with caplog.at_level(logging.ERROR):
        result, error = run_addin_with_logging(addin, None)
    assert result is None
    assert isinstance(error, ValueError)
    assert "fail!" in caplog.text
    assert "FailingAddin failed" in caplog.text
