"""
Groq LLM Provider
=================
Concrete implementation of LLMProvider using the Groq SDK.

Free models available on Groq (as of 2025)
-------------------------------------------
  llama-3.3-70b-versatile   — best quality / reasoning
  llama3-8b-8192            — fastest, lowest latency
  mixtral-8x7b-32768        — long context (32k tokens)
  gemma2-9b-it              — Google Gemma 2

Tool-calling support
---------------------
Groq supports the OpenAI function-calling format.
We convert our LLMMessage list to Groq's format and back.

Retry strategy
--------------
Uses the `backoff` library with exponential back-off for:
  - rate limit errors (429)
  - transient network errors
Max retries and timeout are configurable via settings.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import backoff
from groq import AsyncGroq, APIStatusError, APITimeoutError

from .base_provider import LLMMessage, LLMProvider, LLMResponse

logger = logging.getLogger("mcp_engine.groq")


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception is worth retrying."""
    if isinstance(exc, APIStatusError):
        return exc.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (APITimeoutError, ConnectionError))


class GroqProvider(LLMProvider):
    """
    Async Groq LLM backend.

    Parameters
    ----------
    api_key   : Groq API key (from GROQ_API_KEY env var)
    model     : Groq model ID (default: llama-3.3-70b-versatile)
    timeout   : HTTP timeout in seconds
    max_retries: maximum number of retry attempts
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com"
            )
        self._client = AsyncGroq(api_key=api_key, timeout=timeout, max_retries=0)
        self._model = model
        self._max_retries = max_retries
        logger.info("GroqProvider initialised: model=%s", model)

    @property
    def model_name(self) -> str:
        return self._model

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_groq_messages(messages: list[LLMMessage]) -> list[dict]:
        """Convert internal LLMMessage list to Groq API format."""
        out = []
        for m in messages:
            msg: dict[str, Any] = {"role": m.role, "content": m.content or ""}
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                msg["tool_calls"] = m.tool_calls
                msg["content"] = ""          # Groq requires empty string here
            out.append(msg)
        return out

    @staticmethod
    def _parse_response(resp) -> LLMResponse:
        """Convert a Groq ChatCompletion to LLMResponse."""
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = []
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,  # raw JSON string
                    },
                })

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            model=resp.model,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            } if resp.usage else {},
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Plain text chat completion (no tools)."""
        groq_messages = self._to_groq_messages(messages)
        logger.debug("chat: model=%s messages=%d", self._model, len(messages))

        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=self._max_retries,
            giveup=lambda e: not _is_retryable(e),
            logger=logger,
        )
        async def _call():
            return await self._client.chat.completions.create(
                model=self._model,
                messages=groq_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        resp = await _call()
        result = self._parse_response(resp)
        logger.debug("chat: tokens=%s finish=%s", result.usage, result.finish_reason)
        return result

    async def chat_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """
        Tool-augmented completion.

        Parameters
        ----------
        tools : list of OpenAI-style tool dicts:
            [{
              "type": "function",
              "function": {
                "name": "...",
                "description": "...",
                "parameters": { ... }  # JSON Schema
              }
            }]
        """
        groq_messages = self._to_groq_messages(messages)
        logger.debug(
            "chat_with_tools: model=%s messages=%d tools=%d",
            self._model, len(messages), len(tools),
        )

        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=self._max_retries,
            giveup=lambda e: not _is_retryable(e),
            logger=logger,
        )
        async def _call():
            return await self._client.chat.completions.create(
                model=self._model,
                messages=groq_messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=max_tokens,
                temperature=temperature,
            )

        resp = await _call()
        result = self._parse_response(resp)
        logger.debug(
            "chat_with_tools: finish=%s tool_calls=%d",
            result.finish_reason, len(result.tool_calls),
        )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Factory — create provider from Django settings
# ─────────────────────────────────────────────────────────────────────────────

def get_default_provider() -> GroqProvider:
    """
    Build a GroqProvider from Django settings.

    Usage::

        from mcp_engine.llm.groq_provider import get_default_provider
        llm = get_default_provider()
    """
    from django.conf import settings  # import here to avoid circular at module load

    return GroqProvider(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        timeout=float(settings.GROQ_TIMEOUT),
        max_retries=settings.GROQ_MAX_RETRIES,
    )
