import pytest
from unittest import mock

def test_hello_world():
    assert True

def test_basic_math():
    assert 1 + 1 == 2
    assert 2 * 2 == 4
    assert 5 - 3 == 2

def test_string_manipulation():
    s = "hello world"
    assert s.upper() == "HELLO WORLD"
    assert s.capitalize() == "Hello world"
    assert s.replace("world", "pytest") == "hello pytest"

def test_list_operations():
    l = [1, 2, 3]
    l.append(4)
    assert l == [1, 2, 3, 4]
    l.remove(2)
    assert l == [1, 3, 4]
    assert sum(l) == 8

def test_dict_operations():
    d = {"a": 1, "b": 2}
    d["c"] = 3
    assert d["c"] == 3
    assert set(d.keys()) == {"a", "b", "c"}
    assert d.get("d") is None

def test_mock_example():
    m = mock.Mock()
    m.some_method.return_value = 42
    assert m.some_method() == 42
    m.some_method.assert_called_once()

@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
])
def test_parametrized_addition(a, b, expected):
    assert a + b == expected
