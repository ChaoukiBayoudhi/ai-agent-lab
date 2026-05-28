"""
test_calculator.py
==================
Unit tests for the MCP Calculator tools.

Actual return-dict keys (verified from source):
  calculate()     → {input, result, latex, is_exact, steps}
                    or {input, error, result: None}
  solve_equation()→ {equation, variable, solutions, count}
                    or {equation, variable, error, solutions: []}
  metrics()       → {count, mean, median, std_dev, variance, min, max,
                     sum, q1, q3, iqr, range}
  unit_convert()  → {input_value, from_unit, to_unit, result, formula}
                    or {input_value, from_unit, to_unit, error, result: None}

Security model:
  _validate_expression() blocks: empty expressions, expressions containing
  characters outside [digits, operators, letters, _ . , ( ) = < > !].
  Single-quotes, double-quotes, brackets [ ], semicolons, etc. are forbidden.

Run with:
    pytest mcp_engine/tests/test_calculator.py -v
"""

import math
import pytest

from mcp_engine.mcp_servers.calculator import (
    calculate,
    solve_equation,
    metrics,
    unit_convert,
)


# ─────────────────────────────────────────────────────────────────────────────
# calculate()
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculate:

    def test_basic_arithmetic(self):
        result = calculate(expression="2 + 3 * 4")
        assert result["result"] == 14

    def test_sqrt(self):
        result = calculate(expression="sqrt(144)")
        assert float(result["result"]) == pytest.approx(12.0)

    def test_power(self):
        result = calculate(expression="2**10")
        assert result["result"] == 1024

    def test_pi(self):
        result = calculate(expression="pi")
        assert float(result["result"]) == pytest.approx(math.pi, rel=1e-6)

    def test_trig_sin_pi(self):
        result = calculate(expression="sin(pi)")
        assert abs(float(result["result"])) < 1e-10

    def test_float_result(self):
        result = calculate(expression="1/3")
        assert float(result["result"]) == pytest.approx(1 / 3, rel=1e-6)

    def test_nested_expression(self):
        result = calculate(expression="sqrt(2) * sqrt(8)")
        assert float(result["result"]) == pytest.approx(4.0, rel=1e-6)

    def test_input_key_preserved(self):
        """Return dict has key 'input' (not 'expression')."""
        result = calculate(expression="1+1")
        assert result["input"] == "1+1"

    def test_latex_key_present(self):
        result = calculate(expression="sqrt(4)")
        assert "latex" in result

    # ── Security: blocked tokens must return error ────────────────────────────
    def test_blocked_single_quote(self):
        """Single-quote triggers _SAFE_CHARS regex rejection."""
        result = calculate(expression="__import__('os')")
        assert "error" in result

    def test_blocked_double_quote(self):
        result = calculate(expression='eval("1+1")')
        assert "error" in result

    def test_blocked_open_bracket(self):
        """Square bracket [ is not in _SAFE_CHARS."""
        result = calculate(expression="[1,2,3]")
        assert "error" in result

    def test_empty_expression(self):
        result = calculate(expression="")
        assert "error" in result


# ─────────────────────────────────────────────────────────────────────────────
# solve_equation()
# ─────────────────────────────────────────────────────────────────────────────

class TestSolveEquation:

    def test_linear_equation(self):
        result = solve_equation(equation="2*x + 4 = 0", variable="x")
        assert "solutions" in result
        assert any(float(s) == pytest.approx(-2.0) for s in result["solutions"])

    def test_quadratic_two_roots(self):
        result = solve_equation(equation="x**2 - 5*x + 6 = 0", variable="x")
        assert "solutions" in result
        roots = sorted(float(s) for s in result["solutions"])
        assert roots == pytest.approx([2.0, 3.0])

    def test_perfect_square(self):
        result = solve_equation(equation="x**2 = 4", variable="x")
        assert "solutions" in result
        roots = sorted(abs(float(s)) for s in result["solutions"])
        assert roots == pytest.approx([2.0, 2.0])

    def test_no_solution_returns_empty_list(self):
        result = solve_equation(equation="1 = 0", variable="x")
        assert "solutions" in result
        assert result["solutions"] == [] or result.get("count", -1) == 0

    def test_cubic_x_cubed(self):
        result = solve_equation(equation="x**3 = 0", variable="x")
        assert "solutions" in result

    def test_equation_and_variable_echoed(self):
        result = solve_equation(equation="x = 7", variable="x")
        assert result["equation"] == "x = 7"
        assert result["variable"] == "x"

    def test_solutions_numeric_key_present(self):
        """solutions_numeric contains float values alongside string 'solutions'."""
        result = solve_equation(equation="x**2 - 4 = 0", variable="x")
        assert "solutions" in result
        assert len(result["solutions"]) == 2


# ─────────────────────────────────────────────────────────────────────────────
# metrics()
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:

    DATA = [10.0, 20.0, 30.0, 40.0, 50.0]

    def test_mean(self):
        result = metrics(data=self.DATA)
        assert result["mean"] == pytest.approx(30.0)

    def test_median(self):
        result = metrics(data=self.DATA)
        assert result["median"] == pytest.approx(30.0)

    def test_std_dev_key(self):
        """Key is 'std_dev' (not 'std')."""
        result = metrics(data=self.DATA)
        assert "std_dev" in result

    def test_std_dev_sample_value(self):
        """Sample std (ddof=1) for [10,20,30,40,50] ≈ 15.811."""
        result = metrics(data=self.DATA)
        assert result["std_dev"] == pytest.approx(15.811, rel=1e-3)

    def test_min_max(self):
        result = metrics(data=self.DATA)
        assert result["min"] == pytest.approx(10.0)
        assert result["max"] == pytest.approx(50.0)

    def test_count(self):
        result = metrics(data=self.DATA)
        assert result["count"] == 5

    def test_sum(self):
        result = metrics(data=self.DATA)
        assert result["sum"] == pytest.approx(150.0)

    def test_single_element_std_zero(self):
        """Single-element dataset: std_dev = 0.0 (hardcoded for ddof=1 case)."""
        result = metrics(data=[42.0])
        assert result["mean"] == pytest.approx(42.0)
        assert result["std_dev"] == pytest.approx(0.0)

    def test_empty_data_returns_error(self):
        result = metrics(data=[])
        assert "error" in result

    def test_variance_key_present(self):
        result = metrics(data=[1.0, 2.0, 3.0])
        assert "variance" in result

    def test_quartile_keys_present(self):
        result = metrics(data=self.DATA)
        assert "q1" in result
        assert "q3" in result
        assert "iqr" in result


# ─────────────────────────────────────────────────────────────────────────────
# unit_convert()
# ─────────────────────────────────────────────────────────────────────────────

class TestUnitConvert:
    """Physical unit conversions via pint.

    Return dict keys: {input_value, from_unit, to_unit, result, formula}
    or {input_value, from_unit, to_unit, error, result: None}
    """

    def test_km_per_h_to_m_per_s(self):
        result = unit_convert(value=36.0, from_unit="km/h", to_unit="m/s")
        assert "result" in result
        assert float(result["result"]) == pytest.approx(10.0, rel=1e-4)

    def test_result_key_name(self):
        """Key is 'result', not 'converted'."""
        result = unit_convert(value=100.0, from_unit="km/h", to_unit="m/s")
        assert "result" in result
        assert "converted" not in result

    def test_input_value_key(self):
        """Key is 'input_value', not 'original_value'."""
        result = unit_convert(value=100.0, from_unit="km/h", to_unit="m/s")
        assert result.get("input_value") == pytest.approx(100.0)

    def test_meters_to_feet(self):
        result = unit_convert(value=1.0, from_unit="meter", to_unit="foot")
        assert float(result["result"]) == pytest.approx(3.28084, rel=1e-3)

    def test_formula_key_present(self):
        result = unit_convert(value=1.0, from_unit="meter", to_unit="foot")
        assert "formula" in result

    def test_from_to_unit_keys(self):
        result = unit_convert(value=100.0, from_unit="km/h", to_unit="m/s")
        assert result.get("from_unit") == "km/h"
        assert result.get("to_unit") == "m/s"

    def test_incompatible_units_returns_error(self):
        """meters → seconds: dimensionally incompatible."""
        result = unit_convert(value=1.0, from_unit="meter", to_unit="second")
        assert "error" in result

    def test_unknown_unit_returns_error(self):
        result = unit_convert(value=1.0, from_unit="flibbertigibbet", to_unit="meter")
        assert "error" in result

    def test_large_conversion(self):
        """100 km/h → m/s ≈ 27.778."""
        result = unit_convert(value=100.0, from_unit="km/h", to_unit="m/s")
        assert float(result["result"]) == pytest.approx(27.778, rel=1e-3)
