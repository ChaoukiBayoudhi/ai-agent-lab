"""
Base Agent
==========
Abstract base class for all AI agents in this ecosystem.

Agent Loop (ReAct pattern)
--------------------------
  1. Receive user message
  2. Build prompt with system instructions + conversation history
  3. Call LLM with available tools
  4. If LLM returns tool_calls → execute each tool → add results to history → goto 3
  5. If LLM returns text answer → return final response
  6. Hard-limit: max_steps iterations to prevent infinite loops

Tool execution
--------------
Each agent subclass defines `_tool_definitions()` (OpenAI format) and
`_execute_tool(name, arguments)`.  The base class drives the loop.

A2A integration
---------------
Agents expose `handle_a2a_task(task)` for inter-agent communication.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..llm.base_provider import LLMMessage, LLMProvider, LLMResponse

logger = logging.getLogger("mcp_engine.agent")


@dataclass
class AgentResponse:
    """Structured response returned by every agent."""
    answer: str
    steps: list[dict] = field(default_factory=list)  # reasoning trace
    tool_calls_made: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract agent.

    Parameters
    ----------
    llm         : LLMProvider instance (Groq, OpenAI, …)
    max_steps   : maximum ReAct iterations (default 8)
    """

    def __init__(self, llm: LLMProvider, max_steps: int = 8) -> None:
        self._llm = llm
        self._max_steps = max_steps
        self._name = self.__class__.__name__

    # ── Subclass contract ─────────────────────────────────────────────────────

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System message that shapes the agent's persona and task focus."""
        ...

    @abstractmethod
    def _tool_definitions(self) -> list[dict[str, Any]]:
        """
        Return tools in OpenAI function-calling format:
        [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
        """
        ...

    @abstractmethod
    async def _execute_tool(self, name: str, arguments: dict) -> str:
        """
        Execute tool *name* with *arguments* and return result as a JSON string.
        Raise exceptions for hard failures; return error JSON for soft failures.
        """
        ...

    # ── ReAct loop ────────────────────────────────────────────────────────────

    async def run(self, user_message: str, history: list[LLMMessage] | None = None) -> AgentResponse:
        """
        Run the ReAct loop for *user_message*.

        Parameters
        ----------
        user_message : natural-language question or instruction
        history      : optional prior conversation messages

        Returns
        -------
        AgentResponse with the final answer and reasoning trace.
        """
        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=self.system_prompt),
            *(history or []),
            LLMMessage(role="user", content=user_message),
        ]
        tools = self._tool_definitions()
        steps: list[dict] = []
        tool_calls_made: list[str] = []

        logger.info("[%s] Starting run: %r", self._name, user_message[:80])

        for step in range(1, self._max_steps + 1):
            logger.debug("[%s] Step %d/%d", self._name, step, self._max_steps)

            # ── Call LLM ──────────────────────────────────────────────────────
            if tools:
                resp: LLMResponse = await self._llm.chat_with_tools(messages, tools)
            else:
                resp = await self._llm.chat(messages)

            steps.append({
                "step": step,
                "finish_reason": resp.finish_reason,
                "tool_calls": [tc["function"]["name"] for tc in resp.tool_calls],
                "content_preview": (resp.content or "")[:120],
            })

            # ── No more tool calls → final answer ─────────────────────────────
            if not resp.tool_calls:
                final = resp.content or "No answer generated."
                logger.info("[%s] Final answer after %d step(s).", self._name, step)
                return AgentResponse(
                    answer=final,
                    steps=steps,
                    tool_calls_made=tool_calls_made,
                    metadata={"model": resp.model, "usage": resp.usage, "steps": step},
                )

            # ── Append assistant message with tool_calls ──────────────────────
            messages.append(
                LLMMessage(
                    role="assistant",
                    content=resp.content or "",
                    tool_calls=resp.tool_calls,
                )
            )

            # ── Execute each tool call ────────────────────────────────────────
            for tc in resp.tool_calls:
                fn_name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {"raw": raw_args}

                logger.info("[%s] Tool call: %s(%s)", self._name, fn_name, list(args.keys()))
                tool_calls_made.append(fn_name)

                try:
                    result_str = await self._execute_tool(fn_name, args)
                except Exception as exc:
                    logger.exception("[%s] Tool %s raised: %s", self._name, fn_name, exc)
                    result_str = json.dumps({"error": str(exc)})

                messages.append(
                    LLMMessage(
                        role="tool",
                        content=result_str,
                        tool_call_id=tc["id"],
                    )
                )

        # ── Reached max_steps without a final answer ──────────────────────────
        logger.warning("[%s] Reached max_steps=%d without final answer.", self._name, self._max_steps)
        return AgentResponse(
            answer="I reached the maximum number of reasoning steps without a conclusive answer. Please try a simpler question.",
            steps=steps,
            tool_calls_made=tool_calls_made,
            error="max_steps_exceeded",
        )

    # ── A2A interface ──────────────────────────────────────────────────────────

    async def handle_a2a_task(self, task: dict) -> dict:
        """
        Handle an A2A task delegated by another agent or the coordinator.

        Parameters
        ----------
        task : dict with at least {"task_id": str, "content": {"message": str}}

        Returns
        -------
        dict with {"task_id", "status", "result", "agent"}
        """
        task_id = task.get("task_id", "unknown")
        message = task.get("content", {}).get("message", "")
        logger.info("[%s] Handling A2A task %s", self._name, task_id)

        try:
            resp = await self.run(message)
            return {
                "task_id": task_id,
                "status": "completed",
                "result": resp.answer,
                "tool_calls": resp.tool_calls_made,
                "agent": self._name,
            }
        except Exception as exc:
            logger.exception("[%s] A2A task %s failed: %s", self._name, task_id, exc)
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(exc),
                "agent": self._name,
            }
