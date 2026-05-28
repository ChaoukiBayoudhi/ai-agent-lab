"""
Calculator MCP Server
=====================
Exposes mathematical tools via the MCP protocol (FastMCP variant).

Architecture
------------
  Host (Agent/Django) ──▶ MCP Client ──▶ [this server] ──▶ CalculatorService
                                              JSON-RPC 2.0

Tools exposed
-------------
  calculate       — safe arithmetic / algebraic expression evaluation (sympy)
  solve_equation  — solve one equation for one variable (sympy)
  metrics         — descriptive statistics on a list of numbers (numpy)
  unit_convert    — physical unit conversion (pint)
  matrix_ops      — basic matrix operations (numpy)

Security
--------
  • Expression evaluation uses sympy.sympify() — never Python eval().
  • Input is validated for length and character allow-list before parsing.
  • Division by zero, overflow, and sympy parse errors are caught and
    returned as structured error dicts so the host/agent can handle them.

Running standalone
------------------
  python -m mcp_engine.mcp_servers.calculator   (stdio transport, default)
  python -m mcp_engine.mcp_servers.calculator --port 8001 --transport sse
"""

from __future__ import annotations

import logging
import re
from typing import Annotated

import numpy as np
import sympy as sp
from mcp.server.fastmcp import FastMCP
from pint import UnitRegistry

logger = logging.getLogger("mcp_engine.calculator")

# ─────────────────────────────────────────────────────────────────────────────
# Server instance
# ─────────────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="calculator-server",
    instructions=(
        "A mathematical toolkit. Use `calculate` for arithmetic/algebra, "
        "`solve_equation` for symbolic solving, `metrics` for statistics, "
        "`unit_convert` for physical-unit conversions, "
        "and `matrix_ops` for linear algebra."
    ),
)

# Pint unit registry (singleton — expensive to create)
_ureg = UnitRegistry()

# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

_SAFE_CHARS = re.compile(r"^[\d\s\+\-\*/\^\(\)\.,a-zA-Z_=<>!]+$")
_MAX_EXPR_LEN = 512


def _validate_expression(expr: str) -> None:
    """Raise ValueError if *expr* looks unsafe or malformed."""
    if not expr or not expr.strip():
        raise ValueError("Expression must not be empty.")
    if len(expr) > _MAX_EXPR_LEN:
        raise ValueError(f"Expression too long (max {_MAX_EXPR_LEN} chars).")
    if not _SAFE_CHARS.match(expr):
        raise ValueError(
            "Expression contains disallowed characters. "
            "Only digits, operators (+,-,*,/,^), parentheses, "
            "letters (variables/functions), and common punctuation are allowed."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — calculate
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Evaluate a mathematical expression and return the numeric result. "
        "Supports arithmetic (+,-,*,/,**), trigonometry (sin,cos,tan), "
        "logarithms (log,ln), square root (sqrt), factorial, and symbolic "
        "simplification. Example: 'sqrt(144) + log(E**2)' → 14."
    )
)
def calculate(
    expression: Annotated[str, "Math expression to evaluate, e.g. '2**10 + sqrt(9)'"],
) -> dict:
    """
    Safely evaluate *expression* using sympy.

    Returns
    -------
    dict with keys:
        input      — original expression string
        result     — numeric result (float) or simplified symbolic form (str)
        latex      — LaTeX rendering of the result
        steps      — list of simplification hints (if available)
    """
    logger.debug("calculate: input=%r", expression)
    try:
        _validate_expression(expression)
        parsed = sp.sympify(expression, evaluate=True)
        numeric = float(parsed.evalf()) if parsed.is_number else None  # type: ignore[union-attr]
        result_str = str(numeric) if numeric is not None else str(parsed)

        return {
            "input": expression,
            "result": numeric if numeric is not None else result_str,
            "latex": sp.latex(parsed),
            "is_exact": parsed.is_number and parsed.is_rational,
            "steps": [f"Parsed: {parsed}", f"Evaluated: {result_str}"],
        }
    except (ValueError, sp.SympifyError) as exc:
        logger.warning("calculate error: %s", exc)
        return {"input": expression, "error": str(exc), "result": None}
    except ZeroDivisionError:
        return {"input": expression, "error": "Division by zero.", "result": None}
    except OverflowError:
        return {"input": expression, "error": "Result is too large to represent.", "result": None}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — solve_equation
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Solve an algebraic equation for a given variable using symbolic math. "
        "Equation must use '=' as separator. "
        "Example: equation='x**2 - 5*x + 6 = 0', variable='x' → [2, 3]."
    )
)
def solve_equation(
    equation: Annotated[str, "Equation string with '=', e.g. 'x**2 - 4 = 0'"],
    variable: Annotated[str, "Variable to solve for, e.g. 'x'"] = "x",
) -> dict:
    """
    Solve *equation* for *variable* symbolically.

    Returns
    -------
    dict with keys:
        equation   — original equation string
        variable   — variable solved for
        solutions  — list of solution strings
        latex      — LaTeX rendering of solutions
    """
    logger.debug("solve_equation: eq=%r var=%r", equation, variable)
    try:
        if "=" not in equation:
            raise ValueError("Equation must contain '='.")
        _validate_expression(equation.replace("=", " "))

        lhs_str, rhs_str = equation.split("=", 1)
        lhs = sp.sympify(lhs_str.strip())
        rhs = sp.sympify(rhs_str.strip())
        var = sp.Symbol(variable.strip())
        solutions = sp.solve(lhs - rhs, var)

        return {
            "equation": equation,
            "variable": variable,
            "solutions": [str(s) for s in solutions],
            "solutions_numeric": [float(s.evalf()) for s in solutions if s.is_number],
            "latex": sp.latex(solutions),
        }
    except (ValueError, sp.SympifyError) as exc:
        logger.warning("solve_equation error: %s", exc)
        return {"equation": equation, "variable": variable, "error": str(exc), "solutions": []}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — metrics
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Compute descriptive statistics for a list of numbers. "
        "Returns mean, median, std_dev, variance, min, max, sum, count, "
        "and the 25th/75th percentiles (Q1/Q3)."
    )
)
def metrics(
    data: Annotated[list[float], "List of numeric values, e.g. [1, 2, 3, 4, 5]"],
) -> dict:
    """
    Compute descriptive statistics using NumPy.

    Returns
    -------
    dict of statistical measures. Raises ValueError if *data* is empty.
    """
    logger.debug("metrics: n=%d", len(data))
    if not data:
        return {"error": "Data list must not be empty.", "result": None}
    if len(data) > 10_000:
        return {"error": "Data list too large (max 10 000 items).", "result": None}

    arr = np.array(data, dtype=float)
    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))

    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std_dev": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "variance": float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "sum": float(np.sum(arr)),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "range": float(np.max(arr) - np.min(arr)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 — unit_convert
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Convert a physical quantity from one unit to another using Pint. "
        "Example: value=100, from_unit='km/h', to_unit='m/s' → 27.78."
        "Supports length, mass, time, temperature, energy, pressure, speed, etc."
    )
)
def unit_convert(
    value: Annotated[float, "Numeric value to convert"],
    from_unit: Annotated[str, "Source unit, e.g. 'km/h', 'degC', 'liter'"],
    to_unit: Annotated[str, "Target unit, e.g. 'm/s', 'degF', 'gallon'"],
) -> dict:
    """
    Convert *value from_unit* → *to_unit* using the Pint unit registry.

    Returns
    -------
    dict with input, converted value, and unit strings.
    """
    logger.debug("unit_convert: %s %s → %s", value, from_unit, to_unit)
    try:
        qty = value * _ureg(from_unit)
        converted = qty.to(to_unit)
        return {
            "input_value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "result": round(float(converted.magnitude), 8),
            "formula": f"{value} {from_unit} = {converted.magnitude:.6g} {to_unit}",
        }
    except Exception as exc:  # pint raises various errors
        logger.warning("unit_convert error: %s", exc)
        return {
            "input_value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "error": str(exc),
            "result": None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5 — matrix_ops
# ─────────────────────────────────────────────────────────────────────────────

_MATRIX_OPS = {"add", "subtract", "multiply", "transpose", "determinant", "inverse"}


@mcp.tool(
    description=(
        "Perform matrix operations using NumPy. "
        "Operations: 'add', 'subtract', 'multiply' (dot product), "
        "'transpose', 'determinant', 'inverse'. "
        "Matrices are lists of lists (rows). For unary ops (transpose, det, inv), "
        "provide only matrix_a."
    )
)
def matrix_ops(
    operation: Annotated[str, f"One of: {', '.join(sorted(_MATRIX_OPS))}"],
    matrix_a: Annotated[list[list[float]], "First matrix as list of rows"],
    matrix_b: Annotated[list[list[float]] | None, "Second matrix (for binary ops)"] = None,
) -> dict:
    """
    Execute *operation* on *matrix_a* (and optionally *matrix_b*).

    Returns
    -------
    dict with operation, result matrix or scalar, and shape information.
    """
    logger.debug("matrix_ops: op=%s", operation)
    if operation not in _MATRIX_OPS:
        return {"error": f"Unknown operation '{operation}'. Valid: {sorted(_MATRIX_OPS)}"}

    try:
        A = np.array(matrix_a, dtype=float)
        result: np.ndarray | float

        if operation == "transpose":
            result = A.T
        elif operation == "determinant":
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                return {"error": "Determinant requires a square matrix."}
            result = float(np.linalg.det(A))
        elif operation == "inverse":
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                return {"error": "Inverse requires a square matrix."}
            result = np.linalg.inv(A)
        else:
            if matrix_b is None:
                return {"error": f"Operation '{operation}' requires matrix_b."}
            B = np.array(matrix_b, dtype=float)
            if operation == "add":
                result = A + B
            elif operation == "subtract":
                result = A - B
            elif operation == "multiply":
                result = np.dot(A, B)
            else:
                return {"error": f"Unhandled operation: {operation}"}

        if isinstance(result, np.ndarray):
            return {
                "operation": operation,
                "result": result.tolist(),
                "shape": list(result.shape),
            }
        return {"operation": operation, "result": result}

    except np.linalg.LinAlgError as exc:
        return {"operation": operation, "error": str(exc), "result": None}
    except ValueError as exc:
        return {"operation": operation, "error": str(exc), "result": None}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (standalone process)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculator MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    logger.info("Starting Calculator MCP server (transport=%s)", args.transport)

    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")
