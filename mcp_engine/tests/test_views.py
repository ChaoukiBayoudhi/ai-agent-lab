"""
test_views.py
=============
Integration tests for Django views (HTML pages + JSON API endpoints).

Strategy
--------
* HTML views:  GET → assert 200 + content fragment.
* API views:   POST JSON → mock the orchestrator so no Groq API key needed.
* Orchestrator mock target:
    The views import `from .agents.orchestrator import run as orchestrator_run`
    *inside* the view function body.  Patching works at the module level:
      patch("mcp_engine.agents.orchestrator.run", ...)
* Database:    pytest-django; transaction=True rolls back after each test.

Run with:
    pytest mcp_engine/tests/test_views.py -v
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from django.test import AsyncClient

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_ORCHESTRATOR = "mcp_engine.agents.orchestrator.run"

_MOCK_RESULT = {
    "task_id":    "mock-task-001",
    "agent":      "calculator_agent",
    "answer":     "The result is 42.",
    "tool_calls": ["calculate"],
    "state":      "completed",
    "error":      "",
}


# ─────────────────────────────────────────────────────────────────────────────
# Fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    return AsyncClient()


# ─────────────────────────────────────────────────────────────────────────────
# HTML views — smoke tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_index_200(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert b"AI Agent Lab" in resp.content


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_calculator_ui_200(client):
    resp = await client.get("/calculator/")
    assert resp.status_code == 200
    assert b"Calculator" in resp.content


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_agent_ui_200(client):
    resp = await client.get("/agent/")
    assert resp.status_code == 200
    assert b"Agent" in resp.content


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_a2a_ui_200(client):
    resp = await client.get("/a2a/")
    assert resp.status_code == 200
    assert b"A2A" in resp.content


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_notes_ui_200(client):
    resp = await client.get("/notes/")
    assert resp.status_code == 200
    assert b"Notes" in resp.content


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_history_ui_200(client):
    resp = await client.get("/history/")
    assert resp.status_code == 200
    assert b"History" in resp.content


# ─────────────────────────────────────────────────────────────────────────────
# API — /api/calculate/
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_api_calculate_basic(client):
    resp = await client.post(
        "/api/calculate/",
        data=json.dumps({"tool": "calculate", "params": {"expression": "1 + 1"}}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = json.loads(resp.content)
    assert data["tool"] == "calculate"
    assert "result" in data


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_api_calculate_solve_equation(client):
    resp = await client.post(
        "/api/calculate/",
        data=json.dumps({
            "tool": "solve_equation",
            "params": {"equation": "x**2 - 9 = 0", "variable": "x"},
        }),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = json.loads(resp.content)
    assert "solutions" in data["result"]


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_api_calculate_metrics(client):
    resp = await client.post(
        "/api/calculate/",
        data=json.dumps({
            "tool": "metrics",
            "params": {"data": [1, 2, 3, 4, 5]},
        }),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = json.loads(resp.content)
    assert data["result"]["mean"] == pytest.approx(3.0)


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_api_calculate_unit_convert(client):
    resp = await client.post(
        "/api/calculate/",
        data=json.dumps({
            "tool": "unit_convert",
            "params": {"value": 36.0, "from_unit": "km/h", "to_unit": "m/s"},
        }),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = json.loads(resp.content)
    assert "result" in data["result"]   # key is 'result' not 'converted'


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_api_calculate_missing_params_400(client):
    resp = await client.post(
        "/api/calculate/",
        data=json.dumps({"tool": "calculate", "params": {}}),
        content_type="application/json",
    )
    assert resp.status_code == 400


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_api_calculate_unknown_tool_400(client):
    resp = await client.post(
        "/api/calculate/",
        data=json.dumps({"tool": "unknown_tool", "params": {}}),
        content_type="application/json",
    )
    assert resp.status_code == 400


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_api_calculate_bad_json_400(client):
    resp = await client.post(
        "/api/calculate/",
        data="not-json",
        content_type="application/json",
    )
    assert resp.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# API — /api/agent/chat/
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_api_agent_chat_success(client):
    with patch(_ORCHESTRATOR, new=AsyncMock(return_value=_MOCK_RESULT)):
        resp = await client.post(
            "/api/agent/chat/",
            data=json.dumps({"message": "What is 6 * 7?", "agent": "calculator_agent"}),
            content_type="application/json",
        )
    assert resp.status_code == 200
    data = json.loads(resp.content)
    assert data["answer"] == "The result is 42."
    assert "calculate" in data["tool_calls"]


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_api_agent_chat_missing_message_400(client):
    resp = await client.post(
        "/api/agent/chat/",
        data=json.dumps({"agent": "coordinator"}),
        content_type="application/json",
    )
    assert resp.status_code == 400


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_api_agent_chat_bad_json_400(client):
    resp = await client.post(
        "/api/agent/chat/",
        data="not-json",
        content_type="application/json",
    )
    assert resp.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# API — /api/a2a/delegate/
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_api_a2a_delegate_success(client):
    with patch(_ORCHESTRATOR, new=AsyncMock(return_value=_MOCK_RESULT)):
        resp = await client.post(
            "/api/a2a/delegate/",
            data=json.dumps({
                "message":    "Solve x**2 = 4",
                "from_agent": "planning_agent",
                "to_agent":   "calculator_agent",
            }),
            content_type="application/json",
        )
    assert resp.status_code == 200
    data = json.loads(resp.content)
    assert data["from_agent"] == "planning_agent"
    assert "result" in data
    assert "task_id" in data


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_api_a2a_delegate_missing_message_400(client):
    resp = await client.post(
        "/api/a2a/delegate/",
        data=json.dumps({"from_agent": "user"}),
        content_type="application/json",
    )
    assert resp.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# /api/agents/
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_api_agents_list(client):
    resp = await client.get("/api/agents/")
    assert resp.status_code == 200
    data = json.loads(resp.content)
    assert "agents" in data
    assert isinstance(data["agents"], list)


# ─────────────────────────────────────────────────────────────────────────────
# /.well-known/agent.json
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_well_known_agent_card(client):
    resp = await client.get("/.well-known/agent.json")
    assert resp.status_code == 200
    data = json.loads(resp.content)
    assert data["name"] == "coordinator"
    assert "a2a_endpoint" in data


# ─────────────────────────────────────────────────────────────────────────────
# DB persistence smoke tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_calculation_log_persisted(client):
    """A successful tool call must write a CalculationLog row."""
    from asgiref.sync import sync_to_async
    from mcp_engine.models import CalculationLog

    count_before = await sync_to_async(CalculationLog.objects.count)()
    await client.post(
        "/api/calculate/",
        data=json.dumps({"tool": "calculate", "params": {"expression": "99+1"}}),
        content_type="application/json",
    )
    count_after = await sync_to_async(CalculationLog.objects.count)()
    assert count_after == count_before + 1


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_agent_session_persisted(client):
    """A successful /api/agent/chat/ call must write an AgentSession row."""
    from asgiref.sync import sync_to_async
    from mcp_engine.models import AgentSession

    count_before = await sync_to_async(AgentSession.objects.count)()
    with patch(_ORCHESTRATOR, new=AsyncMock(return_value=_MOCK_RESULT)):
        await client.post(
            "/api/agent/chat/",
            data=json.dumps({"message": "Hello", "agent": "coordinator"}),
            content_type="application/json",
        )
    count_after = await sync_to_async(AgentSession.objects.count)()
    assert count_after == count_before + 1


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_a2a_task_log_persisted(client):
    """A successful /api/a2a/delegate/ call must write an A2ATaskLog row."""
    from asgiref.sync import sync_to_async
    from mcp_engine.models import A2ATaskLog

    count_before = await sync_to_async(A2ATaskLog.objects.count)()
    with patch(_ORCHESTRATOR, new=AsyncMock(return_value=_MOCK_RESULT)):
        await client.post(
            "/api/a2a/delegate/",
            data=json.dumps({
                "message":    "Test task",
                "from_agent": "user",
                "to_agent":   "coordinator",
            }),
            content_type="application/json",
        )
    count_after = await sync_to_async(A2ATaskLog.objects.count)()
    assert count_after == count_before + 1
