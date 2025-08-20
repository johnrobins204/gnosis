import pytest

@pytest.fixture(scope='session')
def sample_fixture():
    return "sample data"

@pytest.fixture
def another_fixture():
    return 42

def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: mark test as smoke test")