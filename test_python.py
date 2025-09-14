# pip install pytest
# pytest -v


def addition(a, b):
    return a + b

#Unit Testing
def test_addition_1():
    a = 2
    b = 3
    expected = 5
    actual = addition(a, b) # 6
    assert actual == expected, f"Expected {a} + {b} to equal {expected}, but got {actual}"

# def test_addition_2():
#     a = -1
#     b = 1
#     expected = 0
#     actual = addition(a, b)

#     assert  actual == expected, "Expected -1 + 1 to equal 0"

