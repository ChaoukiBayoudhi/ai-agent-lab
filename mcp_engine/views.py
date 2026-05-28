"""
Views
=====
Django views for the MCP/Agent lab application.

Structure
---------
  HTML views (render templates):
    index          GET  /
    calculator_ui  GET  /calculator/
    agent_ui       GET  /agent/
    a2a_ui         GET  /a2a/
    notes_ui       GET  /notes/
    history_ui     GET  /history/

  JSON API views (return application/json):
    api_calculate  POST /api/calculate/     — direct MCP calculator call
    api_agent_chat POST /api/agent/chat/    — agent + Groq reasoning
    api_a2a_task   POST /api/a2a/delegate/  — A2A task delegation
    api_agents     GET  /api/agents/        — list agent cards
    api_history    GET  /api/history/       — recent sessions
    agent_card     GET  /.well-known/agent.json  — A2A well-known card

All API views are async (async def) — Django 4.1+ supports native async views.
"""

from __future__ import annotations

import json
import logging
import uuid
from asgiref.sync import sync_to_async

from django.http import JsonResponse, HttpRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .mcp_servers.calculator import calculate, solve_equation, metrics, unit_convert
from .models import CalculationLog, AgentSession, A2ATaskLog

logger = logging.getLogger("mcp_engine.views")


# ─────────────────────────────────────────────────────────────────────────────
# HTML views
# ─────────────────────────────────────────────────────────────────────────────

async def index(request: HttpRequest):
    """Landing page — overview of the ecosystem."""
    from .agents.orchestrator import get_agent_cards
    cards = get_agent_cards()
    return render(request, "mcp_engine/index.html", {"agent_cards": cards})


async def calculator_ui(request: HttpRequest):
    """Calculator playground — call MCP tools directly."""
    recent = await sync_to_async(list)(
        CalculationLog.objects.all()[:10]
    )
    return render(request, "mcp_engine/calculator.html", {"recent": recent})


async def agent_ui(request: HttpRequest):
    """Agent chat interface — talk to CalculatorAgent or NotesAgent."""
    from .agents.orchestrator import get_agent_cards
    cards = get_agent_cards()
    recent = await sync_to_async(list)(
        AgentSession.objects.all()[:10]
    )
    return render(request, "mcp_engine/agent_chat.html", {
        "agent_cards": cards,
        "recent": recent,
    })


async def a2a_ui(request: HttpRequest):
    """A2A demo — visualise task delegation between agents."""
    recent = await sync_to_async(list)(
        A2ATaskLog.objects.all()[:20]
    )
    return render(request, "mcp_engine/a2a_demo.html", {"recent": recent})


async def notes_ui(request: HttpRequest):
    """Notes interface — create and search notes via the Notes agent."""
    recent = await sync_to_async(list)(
        AgentSession.objects.filter(agent_name="notes_agent")[:20]
    )
    return render(request, "mcp_engine/notes.html", {"recent": recent})


async def history_ui(request: HttpRequest):
    """Session history — all past agent runs and calculations."""
    calc_logs    = await sync_to_async(list)(CalculationLog.objects.all()[:100])
    session_logs = await sync_to_async(list)(AgentSession.objects.all()[:100])
    a2a_logs     = await sync_to_async(list)(A2ATaskLog.objects.all()[:100])

    calc_count    = await sync_to_async(CalculationLog.objects.count)()
    session_count = await sync_to_async(AgentSession.objects.count)()
    a2a_count     = await sync_to_async(A2ATaskLog.objects.count)()

    return render(request, "mcp_engine/history.html", {
        "calc_logs":     calc_logs,
        "session_logs":  session_logs,
        "a2a_logs":      a2a_logs,
        "calc_count":    calc_count,
        "session_count": session_count,
        "a2a_count":     a2a_count,
    })


# ─────────────────────────────────────────────────────────────────────────────
# API — Direct Calculator
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["POST"])
async def api_calculate(request: HttpRequest) -> JsonResponse:
    """
    Direct MCP calculator tool call (no LLM).

    POST body (JSON):
        {
          "tool": "calculate" | "solve_equation" | "metrics" | "unit_convert",
          "params": { ... tool-specific params ... }
        }

    Returns: tool result dict + log entry.
    """
    try:
        body = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    tool = body.get("tool", "calculate")
    params = body.get("params", {})

    logger.info("api_calculate: tool=%s", tool)

    # ── Dispatch ──────────────────────────────────────────────────────────────
    result: dict
    if tool == "calculate":
        expr = params.get("expression", "")
        if not expr:
            return JsonResponse({"error": "Missing 'expression' in params."}, status=400)
        result = calculate(expression=expr)
    elif tool == "solve_equation":
        eq = params.get("equation", "")
        if not eq:
            return JsonResponse({"error": "Missing 'equation' in params."}, status=400)
        result = solve_equation(equation=eq, variable=params.get("variable", "x"))
    elif tool == "metrics":
        data = params.get("data", [])
        if not data:
            return JsonResponse({"error": "Missing 'data' list in params."}, status=400)
        result = metrics(data=data)
    elif tool == "unit_convert":
        for k in ("value", "from_unit", "to_unit"):
            if k not in params:
                return JsonResponse({"error": f"Missing '{k}' in params."}, status=400)
        result = unit_convert(**params)
    else:
        return JsonResponse({"error": f"Unknown tool: '{tool}'."}, status=400)

    # ── Persist log ────────────────────────────────────────────────────────────
    success = "error" not in result
    await sync_to_async(CalculationLog.objects.create)(
        expression=json.dumps(params),
        result=json.dumps(result),
        tool_used=tool,
        success=success,
    )

    return JsonResponse({"tool": tool, "params": params, "result": result})


# ─────────────────────────────────────────────────────────────────────────────
# API — Agent Chat
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["POST"])
async def api_agent_chat(request: HttpRequest) -> JsonResponse:
    """
    Send a message to an AI agent (Groq-powered ReAct loop).

    POST body (JSON):
        {
          "message": "What is the determinant of [[1,2],[3,4]]?",
          "agent":   "calculator_agent"   (optional, auto-routed if omitted)
        }

    Returns: answer, agent name, tool calls made, task_id.
    """
    try:
        body = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    message = body.get("message", "").strip()
    if not message:
        return JsonResponse({"error": "Missing 'message' field."}, status=400)

    agent_hint = body.get("agent", "coordinator")

    from .agents.orchestrator import run as orchestrator_run
    result = await orchestrator_run(message=message, agent_hint=agent_hint)

    # ── Persist session ────────────────────────────────────────────────────────
    session_id = result.get("task_id", str(uuid.uuid4())[:12])
    await sync_to_async(AgentSession.objects.create)(
        session_id=session_id,
        user_message=message,
        agent_name=result.get("agent", ""),
        answer=result.get("answer", ""),
        tool_calls=result.get("tool_calls", []),
        error=result.get("error", ""),
    )

    return JsonResponse(result)


# ─────────────────────────────────────────────────────────────────────────────
# API — A2A Task Delegation
# ─────────────────────────────────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["POST"])
async def api_a2a_task(request: HttpRequest) -> JsonResponse:
    """
    Demonstrate A2A by explicitly routing a task through the coordinator.

    POST body (JSON):
        {
          "message":    "Solve x**2 - 4 = 0",
          "from_agent": "planning_agent",    (default "user")
          "to_agent":   "calculator_agent"   (default "coordinator")
        }
    """
    try:
        body = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    message    = body.get("message", "").strip()
    from_agent = body.get("from_agent", "user")
    to_agent   = body.get("to_agent", "coordinator")

    if not message:
        return JsonResponse({"error": "Missing 'message' field."}, status=400)

    from .agents.orchestrator import run as orchestrator_run
    result = await orchestrator_run(
        message=message,
        agent_hint=to_agent,
        from_agent=from_agent,
    )

    # ── Persist A2A log ────────────────────────────────────────────────────────
    await sync_to_async(A2ATaskLog.objects.create)(
        task_id=result.get("task_id", ""),
        from_agent=from_agent,
        to_agent=result.get("agent", to_agent),
        message=message,
        result=result.get("answer", ""),
        state=result.get("state", "completed"),
        tool_calls=result.get("tool_calls", []),
    )

    return JsonResponse({
        "task_id":    result.get("task_id", ""),
        "from_agent": from_agent,
        "to_agent":   result.get("agent", ""),
        "message":    message,
        "result":     result.get("answer", ""),
        "tool_calls": result.get("tool_calls", []),
        "state":      result.get("state", ""),
    })


# ─────────────────────────────────────────────────────────────────────────────
# API — Agent cards
# ─────────────────────────────────────────────────────────────────────────────

async def api_agents(request: HttpRequest) -> JsonResponse:
    """Return all registered agent cards (A2A directory)."""
    from .agents.orchestrator import get_agent_cards
    cards = get_agent_cards()
    return JsonResponse({"agents": cards})


# ─────────────────────────────────────────────────────────────────────────────
# A2A well-known agent card
# ─────────────────────────────────────────────────────────────────────────────

async def agent_well_known(request: HttpRequest) -> JsonResponse:
    """
    Serve the coordinator's Agent Card at /.well-known/agent.json
    (A2A spec well-known URL).
    """
    card = {
        "name": "coordinator",
        "display_name": "AI Agent Lab Coordinator",
        "description": (
            "Multi-agent orchestrator for the MCP/A2A lab. "
            "Routes tasks to the Calculator or Notes agent."
        ),
        "version": "1.0",
        "capabilities": ["orchestration", "routing", "math", "notes"],
        "a2a_endpoint": "/api/a2a/delegate/",
        "agents_directory": "/api/agents/",
    }
    return JsonResponse(card)
