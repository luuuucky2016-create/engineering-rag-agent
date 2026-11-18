from app.tools import calculator


def test_arithmetic_functions_and_constants():
    assert calculator.invoke({"expression": "2 * 18 * 25 * 7.41 / 1000"}) == "6.669"
    assert calculator.invoke({"expression": "sqrt(3) * 400 * 32"}) == "22170.3"
    assert calculator.invoke({"expression": "2^10"}) == "1024"


def test_rejects_code_injection():
    out = calculator.invoke({"expression": "__import__('os').system('echo hacked')"})
    assert out.startswith("Error")


def test_division_by_zero_is_reported_not_raised():
    assert calculator.invoke({"expression": "1 / 0"}) == "Error: division by zero"
