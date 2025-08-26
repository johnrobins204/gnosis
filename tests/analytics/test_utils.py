from src.analytics import utils

def test_utils_safe_divide():
    assert utils.safe_divide(10, 2) == 5
    assert utils.safe_divide(1, 0) == 0
