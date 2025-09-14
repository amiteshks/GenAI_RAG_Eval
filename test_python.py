# pip install pytest
# pytest -v test_python.py
# pytest -v

# Create a divide function
# Create 2 test cases:
    # 1. Test case for two Positive numbers   
    # 2. Test case when Denominator is 0 

def addition(a, b):
    return a + b

#Unit Testing
def test_addition_1():
    a = 2
    b = 3
    expected = 5
    actual = addition(a, b) # 6
    assert actual == expected, f"Expected {a} + {b} to equal {expected}, but got {actual}"



def test_addition_1():
    a = 2
    b = 3
    expected = 5
    actual = addition(a, b) # 6
    assert actual == expected, f"Expected {a} + {b} to equal {expected}, but got {actual}"



def division(a, b):
    if b == 0:
        raise ValueError("Denominator cannot be zero")
    return a / b


def test_division_positive_numbers():
    a = 10
    b = 2
    expected = 5
    actual = division(a, b)
    assert actual == expected, f"Expected {a} / {b} to equal {expected}, but got {actual}"

def test_division_by_zero():
    a = 10
    b = 0
    try:
        division(a, b)
        assert False, "Expected ValueError when dividing by zero, but no error was raised"
    except ValueError as e:
        assert str(e) == "Denominator cannot be zero", f"Expected ValueError message to be 'Denominator cannot be zero', but got '{str(e)}'"    