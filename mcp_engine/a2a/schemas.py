"""
A2A Message Schemas
===================
Defines the data structures used for Agent-to-Agent communication.

Design choices
--------------
We use plain Python dataclasses (not Pydantic) to avoid adding a dependency.
Serialise to/from dict with `to_dict()` / `from_dict()`.

Message types
-------------
  task      — delegate a task to a target agent
  result    — successful completion of a task
  error     — task failed
  status    — probe / heartbeat
  delegate  — coordinator tells an agent to delegate to another

Task states (A2A spec)
---------------------
  submitted → working → completed | failed | canceled
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import uuid


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING   = "working"
    COMPLETED = "completed"
    FAILED    = "failed"
    CANCELED  = "canceled"


class MessageType(str, Enum):
    TASK     = "task"
    RESULT   = "result"
    ERROR    = "error"
    STATUS   = "status"
    DELEGATE = "delegate"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class A2AMessage:
    """
    Core A2A message envelope.

    Fields
    ------
    task_id     : unique ID for this task (auto-generated if not provided)
    message_type: one of MessageType
    from_agent  : name of the sending agent / "user" / "coordinator"
    to_agent    : name of the target agent
    content     : arbitrary payload dict (message, data, result, …)
    state       : current TaskState
    timestamp   : ISO-8601 UTC creation time
    parent_id   : parent task_id for sub-tasks (A2A hierarchical delegation)
    """
    from_agent:   str
    to_agent:     str
    content:      dict[str, Any]
    message_type: MessageType = MessageType.TASK
    task_id:      str         = field(default_factory=lambda: str(uuid.uuid4())[:12])
    state:        TaskState   = TaskState.SUBMITTED
    timestamp:    str         = field(default_factory=_now_iso)
    parent_id:    str | None  = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["message_type"] = self.message_type.value
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "A2AMessage":
        data = dict(data)
        data["message_type"] = MessageType(data.get("message_type", "task"))
        data["state"] = TaskState(data.get("state", "submitted"))
        return cls(**data)

    def make_reply(
        self,
        *,
        result: str,
        state: TaskState = TaskState.COMPLETED,
        from_agent: str,
        extra: dict | None = None,
    ) -> "A2AMessage":
        """Construct a reply message from a task message."""
        return A2AMessage(
            task_id=self.task_id,
            message_type=MessageType.RESULT,
            from_agent=from_agent,
            to_agent=self.from_agent,
            content={"result": result, **(extra or {})},
            state=state,
            parent_id=self.parent_id,
        )

    def make_error(self, error: str, from_agent: str) -> "A2AMessage":
        return A2AMessage(
            task_id=self.task_id,
            message_type=MessageType.ERROR,
            from_agent=from_agent,
            to_agent=self.from_agent,
            content={"error": error},
            state=TaskState.FAILED,
            parent_id=self.parent_id,
        )


@dataclass
class AgentCard:
    """
    Describes an agent's capabilities (A2A well-known format).
    Served at /.well-known/agent.json

    Fields
    ------
    name         : unique agent identifier (snake_case)
    display_name : human-readable label
    description  : what the agent does
    capabilities : list of capability strings ("math", "notes", "search", …)
    skills       : list of example prompts / task descriptions
    """
    name:         str
    display_name: str
    description:  str
    capabilities: list[str]
    skills:       list[str] = field(default_factory=list)
    version:      str = "1.0"

    def to_dict(self) -> dict:
        return asdict(self)
