"""
Orchestrator
============
The single entry point for all AI interactions in the Django app.

Responsibilities
----------------
  • Bootstrap: create LLM provider, instantiate agents, register them
  • Route user messages to the correct agent (via TaskRouter)
  • Expose a simple `async run(message, agent_hint)` API for Django views

Bootstrap is lazy (on first call) so Django start-up is not delayed.
GROQ_API_KEY missing → graceful degradation (returns helpful error message).

Architecture diagram
--------------------

  Django view
      │
      ▼
  Orchestrator.run(message)
      │
      ├─ if GROQ_API_KEY missing → error message
      │
      ├─ TaskRouter.route(A2AMessage)
      │       │
      │       ├─→ CalculatorAgent.run() ─→ Groq LLM ─→ Calculator tools
      │       └─→ NotesAgent.run()      ─→ Groq LLM ─→ Notes tools
      │
      └─ returns AgentResponse dict
"""

from __future__ import annotations

import logging
from typing import Any

from ..a2a.registry import AgentRegistry
from ..a2a.router import TaskRouter
from ..a2a.schemas import A2AMessage, AgentCard, MessageType

logger = logging.getLogger("mcp_engine.orchestrator")

_registry: AgentRegistry | None = None
_router: TaskRouter | None = None
_bootstrapped = False


def _bootstrap() -> None:
    """
    Lazily initialise the LLM provider, agents, and registry.
    Called once on first use.
    """
    global _registry, _router, _bootstrapped
    if _bootstrapped:
        return

    from django.conf import settings
    from ..llm.groq_provider import GroqProvider
    from ..agents.calculator_agent import CalculatorAgent
    from ..agents.notes_agent import NotesAgent

    logger.info("Bootstrapping orchestrator…")

    # ── LLM provider ─────────────────────────────────────────────────────────
    api_key = getattr(settings, "GROQ_API_KEY", "")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — agents will return a setup error.")
        _bootstrapped = True
        return

    llm = GroqProvider(
        api_key=api_key,
        model=settings.GROQ_MODEL,
        timeout=float(settings.GROQ_TIMEOUT),
        max_retries=settings.GROQ_MAX_RETRIES,
    )

    # ── Agents ────────────────────────────────────────────────────────────────
    calc_agent = CalculatorAgent(llm)
    notes_agent = NotesAgent(llm)

    # ── Registry ──────────────────────────────────────────────────────────────
    _registry = AgentRegistry()
    _registry.register(
        calc_agent,
        AgentCard(
            name="calculator_agent",
            display_name="MathBot",
            description="Expert in mathematical calculations, equations, and unit conversions.",
            capabilities=["math", "calculation", "statistics", "unit_conversion"],
            skills=[
                "Solve quadratic equations",
                "Compute matrix determinants",
                "Statistical metrics on datasets",
                "Physical unit conversions",
            ],
        ),
    )
    _registry.register(
        notes_agent,
        AgentCard(
            name="notes_agent",
            display_name="NoteBot",
            description="Manages a personal knowledge base — create, search, and recall notes.",
            capabilities=["notes", "memory", "knowledge_base", "search"],
            skills=[
                "Store a new note",
                "Search existing notes",
                "Retrieve note by ID",
                "List all notes",
            ],
        ),
    )

    _router = TaskRouter(_registry)
    _bootstrapped = True
    logger.info("Orchestrator ready. Agents: %s", _registry.all_names())


async def run(
    message: str,
    agent_hint: str = "coordinator",
    from_agent: str = "user",
) -> dict[str, Any]:
    """
    Main entry point.  Route *message* to the best agent and return result.

    Parameters
    ----------
    message     : User's natural-language message
    agent_hint  : Optional explicit agent name (or "coordinator" for auto-routing)
    from_agent  : Identifier of the caller (default "user")

    Returns
    -------
    dict with keys: answer, agent, tool_calls, task_id, error (if any)
    """
    _bootstrap()

    from django.conf import settings
    if not getattr(settings, "GROQ_API_KEY", ""):
        return {
            "answer": (
                "⚠️  GROQ_API_KEY is not configured. "
                "Copy .env.example to .env, add your key from https://console.groq.com, "
                "and restart the server."
            ),
            "error": "missing_api_key",
            "agent": "none",
        }

    if _router is None or _registry is None:
        return {"answer": "Orchestrator not initialised.", "error": "bootstrap_failed"}

    task = A2AMessage(
        from_agent=from_agent,
        to_agent=agent_hint,
        content={"message": message},
        message_type=MessageType.TASK,
    )

    try:
        result_msg = await _router.route(task)
        return {
            "answer": result_msg.content.get("result", ""),
            "agent": result_msg.content.get("routed_to", result_msg.from_agent),
            "tool_calls": result_msg.content.get("tool_calls", []),
            "task_id": result_msg.task_id,
            "state": result_msg.state.value,
        }
    except Exception as exc:
        logger.exception("Orchestrator run failed: %s", exc)
        return {
            "answer": f"An unexpected error occurred: {exc}",
            "error": str(exc),
            "agent": "orchestrator",
        }


def get_agent_cards() -> list[dict]:
    """Return all registered agent cards (for the UI/API)."""
    _bootstrap()
    if _registry is None:
        return []
    return _registry.all_cards()
