"""
test_a2a.py
===========
Unit tests for the A2A protocol layer.

What is tested
--------------
  A2AMessage   — construction, to_dict / from_dict round-trip, make_reply,
                 make_error, TaskState / MessageType enum values
  AgentRegistry — register / get / find_by_capability / all_cards / reset /
                  singleton identity
  TaskRouter._infer_capability — keyword routing (math vs. notes)

All tests are synchronous (no Groq calls).

Actual signatures (verified from source):
  A2AMessage(from_agent, to_agent, content: dict, message_type, task_id,
             state, timestamp, parent_id)
  make_reply(*, result, state, from_agent, extra=None)  → A2AMessage
  make_error(error, from_agent)                         → A2AMessage

  AgentRegistry()             — singleton via __new__
  registry.register(agent, card)
  registry.get_agent(name)    → BaseAgent | None
  registry.find_by_capability(cap) → list[str] (agent names)
  registry.all_cards()        → list[dict]
  registry.reset()            — clears all registrations

  TaskRouter(registry)._infer_capability(text) → str | None

Run with:
    pytest mcp_engine/tests/test_a2a.py -v
"""

import pytest

from mcp_engine.a2a.schemas import (
    A2AMessage,
    AgentCard,
    TaskState,
    MessageType,
)
from mcp_engine.a2a.registry import AgentRegistry
from mcp_engine.a2a.router import TaskRouter


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_msg(**kwargs) -> A2AMessage:
    """Build an A2AMessage with sensible defaults."""
    defaults = dict(
        task_id="t-001",
        from_agent="user",
        to_agent="coordinator",
        message_type=MessageType.TASK,
        content={"message": "Solve x**2 = 4"},
        state=TaskState.SUBMITTED,
    )
    defaults.update(kwargs)
    return A2AMessage(**defaults)


def _make_card(name: str, capabilities: list[str] | None = None) -> AgentCard:
    return AgentCard(
        name=name,
        display_name=name.replace("_", " ").title(),
        description=f"Test agent {name}",
        capabilities=capabilities or [],
        skills=[],
        version="1.0",
    )


class _DummyAgent:
    """Minimal stub that satisfies AgentRegistry.register(agent, card)."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# A2AMessage
# ─────────────────────────────────────────────────────────────────────────────

class TestA2AMessage:

    def test_instantiation(self):
        msg = _make_msg()
        assert msg.task_id == "t-001"
        assert msg.from_agent == "user"
        assert msg.to_agent == "coordinator"

    def test_content_is_dict(self):
        msg = _make_msg()
        assert isinstance(msg.content, dict)

    def test_to_dict_required_keys(self):
        d = _make_msg().to_dict()
        for key in ("task_id", "from_agent", "to_agent",
                    "message_type", "content", "state"):
            assert key in d, f"Missing key in to_dict(): {key}"

    def test_from_dict_round_trip(self):
        original = _make_msg()
        restored = A2AMessage.from_dict(original.to_dict())
        assert restored.task_id    == original.task_id
        assert restored.from_agent == original.from_agent
        assert restored.content    == original.content
        assert restored.state      == original.state

    def test_make_reply_routing(self):
        request = _make_msg(from_agent="planner", to_agent="coordinator")
        reply = request.make_reply(
            result="Result: ±2",
            state=TaskState.COMPLETED,
            from_agent="coordinator",
        )
        # reply goes back to the original sender
        assert reply.to_agent   == "planner"
        assert reply.from_agent == "coordinator"
        assert reply.task_id    == request.task_id

    def test_make_reply_state(self):
        reply = _make_msg().make_reply(
            result="done",
            state=TaskState.COMPLETED,
            from_agent="agent",
        )
        assert reply.state == TaskState.COMPLETED

    def test_make_reply_preserves_task_id(self):
        msg = _make_msg(task_id="xyz-789")
        reply = msg.make_reply(result="ok", from_agent="a")
        assert reply.task_id == "xyz-789"

    def test_make_error_state_is_failed(self):
        err = _make_msg().make_error(error="No agent found", from_agent="router")
        assert err.state == TaskState.FAILED

    def test_make_error_content_has_error(self):
        err = _make_msg().make_error(error="Timeout", from_agent="router")
        assert "Timeout" in err.content.get("error", "")

    def test_make_error_routing(self):
        msg = _make_msg(from_agent="sender")
        err = msg.make_error(error="fail", from_agent="router")
        assert err.to_agent == "sender"

    def test_task_state_enum_string_values(self):
        assert TaskState.SUBMITTED.value == "submitted"
        assert TaskState.COMPLETED.value == "completed"
        assert TaskState.FAILED.value    == "failed"
        assert TaskState.WORKING.value   == "working"
        assert TaskState.CANCELED.value  == "canceled"

    def test_message_type_enum_string_values(self):
        assert MessageType.TASK.value   == "task"
        assert MessageType.RESULT.value == "result"
        assert MessageType.ERROR.value  == "error"


# ─────────────────────────────────────────────────────────────────────────────
# AgentRegistry
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentRegistry:
    """Each test method gets a fresh registry via the autouse fixture."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        reg = AgentRegistry()
        reg.reset()
        yield reg
        reg.reset()

    def test_singleton_same_instance(self):
        reg1 = AgentRegistry()
        reg2 = AgentRegistry()
        assert reg1 is reg2

    def test_register_and_get_card(self, reset_registry):
        card = _make_card("calc_agent", ["math"])
        reset_registry.register(_DummyAgent(), card)
        retrieved = reset_registry.get_card("calc_agent")
        assert retrieved is not None
        assert retrieved.name == "calc_agent"

    def test_get_agent_returns_instance(self, reset_registry):
        agent = _DummyAgent()
        card = _make_card("mybot")
        reset_registry.register(agent, card)
        assert reset_registry.get_agent("mybot") is agent

    def test_get_unknown_agent_returns_none(self, reset_registry):
        assert reset_registry.get_agent("ghost_agent") is None

    def test_all_cards_empty(self, reset_registry):
        assert reset_registry.all_cards() == []

    def test_all_cards_populated(self, reset_registry):
        reset_registry.register(_DummyAgent(), _make_card("a"))
        reset_registry.register(_DummyAgent(), _make_card("b"))
        assert len(reset_registry.all_cards()) == 2

    def test_register_duplicate_overwrites(self, reset_registry):
        reset_registry.register(_DummyAgent(), _make_card("dup", ["old"]))
        reset_registry.register(_DummyAgent(), _make_card("dup", ["new"]))
        assert "new" in reset_registry.get_card("dup").capabilities

    def test_find_by_capability_found(self, reset_registry):
        reset_registry.register(_DummyAgent(), _make_card("math_bot", ["math", "calc"]))
        reset_registry.register(_DummyAgent(), _make_card("note_bot", ["notes"]))
        names = reset_registry.find_by_capability("math")
        assert "math_bot" in names
        assert "note_bot" not in names

    def test_find_by_capability_not_found(self, reset_registry):
        reset_registry.register(_DummyAgent(), _make_card("math_bot", ["math"]))
        assert reset_registry.find_by_capability("cooking") == []

    def test_reset_clears_all(self, reset_registry):
        reset_registry.register(_DummyAgent(), _make_card("a"))
        reset_registry.register(_DummyAgent(), _make_card("b"))
        reset_registry.reset()
        assert reset_registry.all_cards() == []


# ─────────────────────────────────────────────────────────────────────────────
# TaskRouter._infer_capability (keyword routing)
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskRouterInferCapability:
    """
    Tests for the keyword-based routing rules via _infer_capability().
    We test the private helper directly — no async needed, no actual agents.
    """

    @pytest.fixture()
    def router(self):
        # TaskRouter requires a registry, but _infer_capability doesn't use it
        return TaskRouter(AgentRegistry())

    # ── Math routing ──────────────────────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "Calculate the sum of 1 and 2",
        "solve x**2 - 4 = 0",
        "What is the determinant of this matrix?",
        "arithmetic mean of [1,2,3]",
        "compute sqrt(2)",
        "statistics of [10,20,30]",
        "convert 100 km/h to m/s",
        "solve the equation x = 5",
        "What are the metrics of this dataset?",
        "eigenvalue of A",
    ])
    def test_infers_math(self, router, text):
        cap = router._infer_capability(text)
        assert cap == "math", f"Expected 'math' for: {text!r}, got {cap!r}"

    # ── Notes routing ─────────────────────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "Remember that Python uses duck typing",
        "save this note: ASGI supports WebSockets",
        "Find my note about Django",
        "search notes for asyncio",
        "write down: the event loop is single-threaded",
        "recall my notes",
    ])
    def test_infers_notes(self, router, text):
        cap = router._infer_capability(text)
        assert cap == "notes", f"Expected 'notes' for: {text!r}, got {cap!r}"

    # ── Unrecognised → None ───────────────────────────────────────────────────
    def test_unrecognized_returns_none(self, router):
        cap = router._infer_capability("Tell me a joke about penguins")
        assert cap is None

    def test_case_insensitive_math(self, router):
        assert router._infer_capability("CALCULATE 1+1") == "math"

    def test_case_insensitive_notes(self, router):
        assert router._infer_capability("REMEMBER this") == "notes"
