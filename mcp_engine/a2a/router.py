"""
Task Router
===========
Routes incoming A2A messages to the appropriate agent based on:
  1. Explicit `to_agent` in the message
  2. Capability matching (keyword routing)
  3. LLM-based routing (fallback, if enabled)

Routing rules (keyword-based)
------------------------------
  Keywords in the message → Agent capability
  "calculat", "math", "equation", "solv", "convert", "matrix", "statistic"
      → "math"
  "note", "remember", "store", "find", "search note", "recall"
      → "notes"

The router is intentionally simple and extensible — add new rules to
_ROUTING_RULES without changing calling code.
"""

from __future__ import annotations

import logging
import re

from .registry import AgentRegistry
from .schemas import A2AMessage, TaskState

logger = logging.getLogger("mcp_engine.router")


# ── Routing rules: (regex_pattern, capability) ──────────────────────────────

_ROUTING_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"calculat|arith|math|equat|solv|sqrt|matrix|determin|eigen"
                r"|statistic|metric|mean|median|std|variance|unit convert|km/h"
                r"|newton|joule|pascal", re.I), "math"),
    (re.compile(r"\bnote\b|remember|store.*note|find.*note|search.*note"
                r"|recall|my note|write down|save this", re.I), "notes"),
]


class TaskRouter:
    """
    Routes A2AMessage objects to the correct registered agent.

    Usage::

        router = TaskRouter(registry)
        result_message = await router.route(a2a_message)
    """

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    def _infer_capability(self, text: str) -> str | None:
        """Return the best-matching capability string, or None."""
        for pattern, capability in _ROUTING_RULES:
            if pattern.search(text):
                return capability
        return None

    def _select_agent_name(self, message: A2AMessage) -> str | None:
        """
        Determine which agent should handle *message*.

        Priority
        --------
        1. Explicit `to_agent` field (if that agent is registered)
        2. Capability matching on message content
        3. First registered agent (last-resort fallback)
        """
        # 1. Explicit target
        if message.to_agent and message.to_agent != "coordinator":
            if self._registry.get_agent(message.to_agent):
                return message.to_agent

        # 2. Capability routing
        text = message.content.get("message", "")
        capability = self._infer_capability(text)
        if capability:
            candidates = self._registry.find_by_capability(capability)
            if candidates:
                return candidates[0]

        # 3. Fallback to first agent
        names = self._registry.all_names()
        return names[0] if names else None

    async def route(self, message: A2AMessage) -> A2AMessage:
        """
        Route *message* and return the result A2AMessage.

        If no suitable agent is found, returns an error message.
        """
        agent_name = self._select_agent_name(message)
        if not agent_name:
            logger.warning("No agent found for message: %s", message.task_id)
            return message.make_error("No suitable agent found.", from_agent="router")

        agent = self._registry.get_agent(agent_name)
        logger.info(
            "Routing task %s → %s (capability=%s)",
            message.task_id, agent_name,
            self._infer_capability(message.content.get("message", "")),
        )

        result = await agent.handle_a2a_task(message.to_dict())

        return A2AMessage(
            task_id=message.task_id,
            message_type=__import__("mcp_engine.a2a.schemas", fromlist=["MessageType"]).MessageType.RESULT,
            from_agent=agent_name,
            to_agent=message.from_agent,
            content={
                "result": result.get("result", ""),
                "tool_calls": result.get("tool_calls", []),
                "routed_to": agent_name,
            },
            state=TaskState.COMPLETED if result.get("status") == "completed" else TaskState.FAILED,
            parent_id=message.parent_id,
        )
