"""
Abstract LLM Provider
=====================
Defines the contract that every LLM backend must fulfill.
This abstraction lets us swap Groq → OpenAI → local Ollama
by changing a single environment variable.

Pattern: Strategy — the agent holds a reference to an LLMProvider instance
and calls `chat()` / `chat_with_tools()` without knowing the backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMMessage:
    """Single message in a conversation."""
    role: str          # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: str | None = None    # populated for role="tool" responses
    tool_calls: list[dict] | None = None  # populated when assistant invokes tools


@dataclass
class LLMResponse:
    """What the LLM returned."""
    content: str | None
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"     # "stop" | "tool_calls" | "length"
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Subclasses must implement:
      - chat()            — plain text completion
      - chat_with_tools() — completion with tool-calling support
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Send *messages* and return the assistant reply."""
        ...

    @abstractmethod
    async def chat_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """
        Send *messages* with *tools* (OpenAI function-calling format).
        Returns either a text reply or a list of tool_calls to execute.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...
