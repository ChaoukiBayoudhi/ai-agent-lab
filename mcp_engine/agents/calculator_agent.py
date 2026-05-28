"""
Calculator Agent
================
An AI agent that uses the Calculator MCP server to answer
mathematical questions.

How it works
------------
  1. User asks a question (e.g. "What is the eigenvalue of [[1,2],[3,4]]?")
  2. Agent asks Groq LLM (ReAct loop, see BaseAgent)
  3. LLM decides which tool to call (calculate / solve_equation / …)
  4. Agent invokes the tool directly on the in-process MCP server module
     (no subprocess needed — we import the FastMCP instance and call tools)
  5. Tool result is fed back to LLM
  6. LLM produces final answer

In-process tool execution
-------------------------
Rather than spawning a subprocess and using stdio transport,
we call the MCP tool functions directly via Python imports.
This is the simplest integration pattern for same-process deployments.
For production multi-tenant use, start the server as a subprocess and
use httpx + SSE transport.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .base_agent import AgentResponse, BaseAgent
from ..llm.base_provider import LLMProvider
from ..mcp_servers.calculator import (
    calculate,
    matrix_ops,
    metrics,
    solve_equation,
    unit_convert,
)

logger = logging.getLogger("mcp_engine.calculator_agent")

# ─────────────────────────────────────────────────────────────────────────────
# Tool definitions (OpenAI / Groq function-calling format)
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a mathematical expression. Supports arithmetic, "
                "trigonometry (sin, cos, tan), logarithms (log, ln), sqrt, "
                "exponents, and symbolic algebra."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. 'sin(pi/2) + sqrt(16)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_equation",
            "description": (
                "Solve an algebraic equation for a variable. "
                "Equation must contain '='. Example: 'x**2 - 5*x + 6 = 0'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {
                        "type": "string",
                        "description": "Equation string with '=', e.g. '2*x + 3 = 7'",
                    },
                    "variable": {
                        "type": "string",
                        "description": "Variable to solve for (default 'x')",
                        "default": "x",
                    },
                },
                "required": ["equation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "metrics",
            "description": "Compute descriptive statistics (mean, median, std, …) for a list of numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of numeric values",
                    }
                },
                "required": ["data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_convert",
            "description": (
                "Convert a physical quantity between units. "
                "E.g. 100 km/h → m/s, 0 degC → degF."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Numeric value"},
                    "from_unit": {"type": "string", "description": "Source unit string"},
                    "to_unit": {"type": "string", "description": "Target unit string"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "matrix_ops",
            "description": (
                "Matrix operations: add, subtract, multiply (dot), "
                "transpose, determinant, inverse."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "transpose", "determinant", "inverse"],
                    },
                    "matrix_a": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "First matrix as list of rows",
                    },
                    "matrix_b": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Second matrix (required for add/subtract/multiply)",
                    },
                },
                "required": ["operation", "matrix_a"],
            },
        },
    },
]


class CalculatorAgent(BaseAgent):
    """
    Agent specialised in mathematical reasoning.
    Uses the Calculator MCP server tools in-process.
    """

    SYSTEM_PROMPT = (
        "You are MathBot, an expert AI assistant for mathematical calculations. "
        "You have access to powerful math tools. Always use the appropriate tool "
        "to compute results — do NOT guess numerical answers. "
        "When you have the tool result, explain it clearly in plain language. "
        "Show your reasoning step by step. "
        "If the question is ambiguous, state your assumption before calculating."
    )

    def __init__(self, llm: LLMProvider, max_steps: int = 8) -> None:
        super().__init__(llm, max_steps)

    @property
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return _TOOLS

    async def _execute_tool(self, name: str, arguments: dict) -> str:
        """
        Dispatch to the appropriate Calculator MCP tool function.
        All tool functions are synchronous — we call them directly.
        """
        logger.info("Executing calculator tool: %s(%s)", name, list(arguments.keys()))

        if name == "calculate":
            result = calculate(expression=arguments["expression"])
        elif name == "solve_equation":
            result = solve_equation(
                equation=arguments["equation"],
                variable=arguments.get("variable", "x"),
            )
        elif name == "metrics":
            result = metrics(data=arguments["data"])
        elif name == "unit_convert":
            result = unit_convert(
                value=arguments["value"],
                from_unit=arguments["from_unit"],
                to_unit=arguments["to_unit"],
            )
        elif name == "matrix_ops":
            result = matrix_ops(
                operation=arguments["operation"],
                matrix_a=arguments["matrix_a"],
                matrix_b=arguments.get("matrix_b"),
            )
        else:
            result = {"error": f"Unknown tool: {name}"}

        return json.dumps(result, default=str)
