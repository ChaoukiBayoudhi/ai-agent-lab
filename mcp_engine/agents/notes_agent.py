"""
Notes Agent
===========
An AI agent that manages a personal knowledge base using the Notes MCP server.

Use cases
---------
  - "Remember that Paris is the capital of France"
  - "Store my homework deadline for next Monday"
  - "Find my note about Python asyncio"
  - "What notes do I have tagged 'algorithm'?"

The agent understands natural language instructions and maps them
to the appropriate notes tool calls.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .base_agent import BaseAgent
from ..llm.base_provider import LLMProvider
from ..mcp_servers.notes import (
    create_note,
    delete_note,
    get_note,
    list_notes,
    search_notes,
    update_note,
)

logger = logging.getLogger("mcp_engine.notes_agent")

# ─────────────────────────────────────────────────────────────────────────────
# Tool definitions
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "create_note",
            "description": "Create and store a new note with a title and body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title for the note"},
                    "body": {"type": "string", "description": "Content / body of the note (markdown)"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags for categorisation",
                    },
                },
                "required": ["title", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_note",
            "description": "Retrieve a note by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {"type": "string", "description": "The 8-character note ID"},
                },
                "required": ["note_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": "List all stored notes (titles and IDs).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Search notes by keyword (title, body, or tags).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_note",
            "description": "Update an existing note's title or body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {"type": "string"},
                    "title": {"type": "string", "description": "New title (optional)"},
                    "body": {"type": "string", "description": "New body (optional)"},
                },
                "required": ["note_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_note",
            "description": "Delete a note by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {"type": "string"},
                },
                "required": ["note_id"],
            },
        },
    },
]


class NotesAgent(BaseAgent):
    """
    Agent specialised in note management.
    Uses the Notes MCP server tools in-process.
    """

    SYSTEM_PROMPT = (
        "You are NoteBot, an AI personal assistant for managing notes and information. "
        "When a user asks you to remember something, create a well-structured note. "
        "When they ask to find information, search the notes. "
        "Always confirm what you stored or found. "
        "Use descriptive titles and add relevant tags (e.g. ['python', 'algorithm']). "
        "If the user asks to list or search notes, show clear summaries."
    )

    def __init__(self, llm: LLMProvider, max_steps: int = 6) -> None:
        super().__init__(llm, max_steps)

    @property
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return _TOOLS

    async def _execute_tool(self, name: str, arguments: dict) -> str:
        logger.info("Executing notes tool: %s(%s)", name, list(arguments.keys()))

        if name == "create_note":
            result = create_note(
                title=arguments["title"],
                body=arguments["body"],
                tags=arguments.get("tags"),
            )
        elif name == "get_note":
            result = get_note(note_id=arguments["note_id"])
        elif name == "list_notes":
            result = list_notes()
        elif name == "search_notes":
            result = search_notes(query=arguments["query"])
        elif name == "update_note":
            result = update_note(
                note_id=arguments["note_id"],
                title=arguments.get("title"),
                body=arguments.get("body"),
            )
        elif name == "delete_note":
            result = delete_note(note_id=arguments["note_id"])
        else:
            result = {"error": f"Unknown tool: {name}"}

        return json.dumps(result, default=str)
