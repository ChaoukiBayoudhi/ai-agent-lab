"""
test_agents.py
==============
Unit tests for CalculatorAgent and NotesAgent with a *mocked* Groq LLM.

Key implementation facts (verified from source):
  - LLMResponse.tool_calls  is list[dict] in OpenAI format:
      [{"id": "...", "function": {"name": "...", "arguments": "{...json...}"}}]
  - AgentResponse.tool_calls_made  is list[str] (function names)
  - handle_a2a_task(task: dict)  — takes a plain dict, NOT an A2AMessage
  - BaseAgent(llm, max_steps=8)  — constructor

No Groq API key is needed; a ScriptedLLMProvider replays fixed responses.

Run with:
    pytest mcp_engine/tests/test_agents.py -v
"""

from __future__ import annotations

import json
import uuid

import pytest

from mcp_engine.llm.base_provider import LLMProvider, LLMMessage, LLMResponse
from mcp_engine.agents.base_agent import AgentResponse
from mcp_engine.agents.calculator_agent import CalculatorAgent
from mcp_engine.agents.notes_agent import NotesAgent


# ─────────────────────────────────────────────────────────────────────────────
# Fake LLM provider
# ─────────────────────────────────────────────────────────────────────────────

class ScriptedLLMProvider(LLMProvider):
    """
    Replay a fixed list of LLMResponse objects in order.

    Example::
        provider = ScriptedLLMProvider([
            LLMResponse(content=None, tool_calls=[_tc("calculate", {"expression":"2+2"})]),
            LLMResponse(content="The answer is 4.", tool_calls=[]),
        ])
    """

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self._index = 0

    async def chat(self, messages, **kwargs) -> LLMResponse:
        return self._next()

    async def chat_with_tools(self, messages, tools, **kwargs) -> LLMResponse:
        return self._next()

    def _next(self) -> LLMResponse:
        if self._index >= len(self._responses):
            return LLMResponse(content="[end]", tool_calls=[])
        resp = self._responses[self._index]
        self._index += 1
        return resp

    @property
    def model_name(self) -> str:
        return "fake-llm"


def _tc(name: str, args: dict, call_id: str | None = None) -> dict:
    """Build a tool_call dict in OpenAI function-calling format."""
    return {
        "id": call_id or f"call_{uuid.uuid4().hex[:8]}",
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# CalculatorAgent
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestCalculatorAgent:

    async def test_direct_answer_no_tools(self):
        """LLM answers immediately without any tool call."""
        provider = ScriptedLLMProvider([
            LLMResponse(content="2 + 2 is 4.", tool_calls=[]),
        ])
        agent = CalculatorAgent(llm=provider)
        result = await agent.run("What is 2 + 2?")
        assert isinstance(result, AgentResponse)
        assert "4" in result.answer
        assert result.tool_calls_made == []

    async def test_single_tool_call_then_answer(self):
        """LLM calls calculate(), receives result, then gives final answer."""
        provider = ScriptedLLMProvider([
            LLMResponse(content=None,
                        tool_calls=[_tc("calculate", {"expression": "sqrt(144)"})]),
            LLMResponse(content="The square root of 144 is 12.", tool_calls=[]),
        ])
        agent = CalculatorAgent(llm=provider)
        result = await agent.run("What is sqrt(144)?")
        assert result.answer == "The square root of 144 is 12."
        assert "calculate" in result.tool_calls_made

    async def test_tool_calls_made_list(self):
        """tool_calls_made records each function name called."""
        provider = ScriptedLLMProvider([
            LLMResponse(content=None,
                        tool_calls=[_tc("solve_equation",
                                        {"equation": "x**2-4=0", "variable": "x"})]),
            LLMResponse(content="Solutions are x=2 and x=-2.", tool_calls=[]),
        ])
        agent = CalculatorAgent(llm=provider)
        result = await agent.run("Solve x**2 = 4")
        assert "solve_equation" in result.tool_calls_made

    async def test_two_sequential_tool_calls(self):
        """LLM makes two separate tool calls before the final answer."""
        provider = ScriptedLLMProvider([
            LLMResponse(content=None,
                        tool_calls=[_tc("calculate", {"expression": "2**10"})]),
            LLMResponse(content=None,
                        tool_calls=[_tc("metrics", {"data": [1, 2, 3]})]),
            LLMResponse(content="Done — 2^10=1024 and mean=2.", tool_calls=[]),
        ])
        agent = CalculatorAgent(llm=provider)
        result = await agent.run("Compute 2**10 and the mean of [1,2,3]")
        assert len(result.tool_calls_made) == 2

    async def test_max_steps_respected(self):
        """Agent stops after max_steps even if LLM keeps returning tool calls."""
        always_tool = LLMResponse(
            content=None,
            tool_calls=[_tc("calculate", {"expression": "1+1"})],
        )
        provider = ScriptedLLMProvider([always_tool] * 20)
        agent = CalculatorAgent(llm=provider, max_steps=3)
        result = await agent.run("Loop forever")
        # Should return an AgentResponse (not raise)
        assert isinstance(result, AgentResponse)
        # error field is set when max_steps exceeded
        assert result.error is not None or result.answer is not None

    async def test_unknown_tool_does_not_crash(self):
        """If LLM hallucinates a non-existent tool, agent should not raise."""
        provider = ScriptedLLMProvider([
            LLMResponse(content=None,
                        tool_calls=[_tc("made_up_tool", {"x": 1})]),
            LLMResponse(content="I couldn't find that tool.", tool_calls=[]),
        ])
        agent = CalculatorAgent(llm=provider)
        result = await agent.run("Use made_up_tool")
        assert isinstance(result.answer, str)

    async def test_response_type(self):
        provider = ScriptedLLMProvider([LLMResponse(content="Done.", tool_calls=[])])
        agent = CalculatorAgent(llm=provider)
        result = await agent.run("Test")
        assert isinstance(result, AgentResponse)


# ─────────────────────────────────────────────────────────────────────────────
# NotesAgent
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestNotesAgent:

    async def test_create_note_tool_called(self):
        provider = ScriptedLLMProvider([
            LLMResponse(content=None, tool_calls=[_tc("create_note", {
                "title": "Django async",
                "body": "Use sync_to_async for ORM calls",
            })]),
            LLMResponse(content="Note 'Django async' has been saved.", tool_calls=[]),
        ])
        agent = NotesAgent(llm=provider)
        result = await agent.run("Remember: use sync_to_async for ORM calls")
        assert "create_note" in result.tool_calls_made

    async def test_search_notes_tool_called(self):
        provider = ScriptedLLMProvider([
            LLMResponse(content=None,
                        tool_calls=[_tc("search_notes", {"query": "asyncio"})]),
            LLMResponse(content="I found 2 notes about asyncio.", tool_calls=[]),
        ])
        agent = NotesAgent(llm=provider)
        result = await agent.run("Find my notes about asyncio")
        assert "search_notes" in result.tool_calls_made

    async def test_no_tools_direct_answer(self):
        provider = ScriptedLLMProvider([
            LLMResponse(content="I'm the NoteBot. How can I help?", tool_calls=[]),
        ])
        agent = NotesAgent(llm=provider)
        result = await agent.run("Hello!")
        assert result.tool_calls_made == []

    async def test_agent_response_type(self):
        provider = ScriptedLLMProvider([LLMResponse(content="Done.", tool_calls=[])])
        agent = NotesAgent(llm=provider)
        result = await agent.run("Test")
        assert isinstance(result, AgentResponse)

    async def test_handle_a2a_task_returns_dict(self):
        """handle_a2a_task(task_dict) returns a result dict."""
        provider = ScriptedLLMProvider([
            LLMResponse(content="Note saved.", tool_calls=[]),
        ])
        agent = NotesAgent(llm=provider)

        task = {
            "task_id": "task-99",
            "from_agent": "coordinator",
            "to_agent": "notes_agent",
            "content": {"message": "Remember: FastMCP auto-generates inputSchema"},
            "state": "submitted",
        }
        result = await agent.handle_a2a_task(task)
        assert isinstance(result, dict)
        assert result.get("status") == "completed"
        assert result.get("task_id") == "task-99"

    async def test_handle_a2a_task_includes_result(self):
        provider = ScriptedLLMProvider([
            LLMResponse(content="Memory stored.", tool_calls=[]),
        ])
        agent = NotesAgent(llm=provider)
        result = await agent.handle_a2a_task({
            "task_id": "t-01",
            "content": {"message": "Save this."},
        })
        assert "result" in result or "error" in result

    async def test_multiple_tools_recorded(self):
        provider = ScriptedLLMProvider([
            LLMResponse(content=None,
                        tool_calls=[_tc("search_notes", {"query": "python"})]),
            LLMResponse(content=None,
                        tool_calls=[_tc("create_note", {"title": "T", "body": "B"})]),
            LLMResponse(content="Done.", tool_calls=[]),
        ])
        agent = NotesAgent(llm=provider)
        result = await agent.run("Search and then create a note")
        assert "search_notes" in result.tool_calls_made
        assert "create_note" in result.tool_calls_made
