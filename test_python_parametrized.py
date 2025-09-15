# pytest -v
# pip install pytest
import pytest

def addition(a, b):
    return a + b

[(10, 5, 15), # a = 10, b = 5, expected = 15
 (0, 0, 0)]  # a = 0, b = 0, expected = 0

test_data = [(10, 5, 15),(0, 0, 0)]

# Annotate the test function with @pytest.mark.parametrize
@pytest.mark.parametrize("a,b,expected", test_data)
def test_addition_parametrized(a, b, expected):
    actual = addition(a, b)
    assert  actual == expected, f"Expected {a} + {b} to equal {expected}"