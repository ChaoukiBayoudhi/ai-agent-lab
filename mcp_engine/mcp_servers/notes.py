"""
Notes MCP Server
================
A simple but practical in-process note-taking server — no external services.

Why a Notes Server?
-------------------
It demonstrates that MCP is not only for computation: any structured
data service (documents, KV store, search index) can be wrapped as an
MCP server.  Agents can use it to *remember* information between turns,
store intermediate reasoning, or accumulate a research log.

Architecture
------------
  Agent / Django ──▶ MCP Client ──▶ [this server] ──▶ NoteStore (dict)

Tools exposed
-------------
  create_note   — add a new note with a title and markdown body
  get_note      — retrieve a single note by ID
  list_notes    — list all note IDs and titles
  search_notes  — full-text search across titles and bodies
  delete_note   — remove a note by ID
  update_note   — edit the title or body of an existing note

Storage
-------
In-memory dict (per-process).  Restarting the server clears all notes.
For persistence, swap NoteStore for a SQLite or JSON-file backend.

Running standalone
------------------
  python -m mcp_engine.mcp_servers.notes
  python -m mcp_engine.mcp_servers.notes --port 8002 --transport sse
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("mcp_engine.notes")

# ─────────────────────────────────────────────────────────────────────────────
# Server instance
# ─────────────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="notes-server",
    instructions=(
        "A note-taking service. Use `create_note` to store information, "
        "`search_notes` to find relevant notes, and `list_notes` to see all notes."
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# In-memory store
# ─────────────────────────────────────────────────────────────────────────────

class NoteStore:
    """Thread-safe (for MCP's async context) in-memory note repository."""

    def __init__(self) -> None:
        self._notes: dict[str, dict] = {}

    def create(self, title: str, body: str, tags: list[str] | None = None) -> dict:
        note_id = str(uuid.uuid4())[:8]  # short ID for readability
        now = datetime.now(timezone.utc).isoformat()
        note = {
            "id": note_id,
            "title": title.strip(),
            "body": body.strip(),
            "tags": tags or [],
            "created_at": now,
            "updated_at": now,
        }
        self._notes[note_id] = note
        logger.info("Note created: id=%s title=%r", note_id, title)
        return note

    def get(self, note_id: str) -> dict | None:
        return self._notes.get(note_id)

    def update(self, note_id: str, title: str | None, body: str | None) -> dict | None:
        note = self._notes.get(note_id)
        if note is None:
            return None
        if title is not None:
            note["title"] = title.strip()
        if body is not None:
            note["body"] = body.strip()
        note["updated_at"] = datetime.now(timezone.utc).isoformat()
        return note

    def delete(self, note_id: str) -> bool:
        return self._notes.pop(note_id, None) is not None

    def list_all(self) -> list[dict]:
        return [
            {"id": n["id"], "title": n["title"], "tags": n["tags"],
             "created_at": n["created_at"]}
            for n in sorted(self._notes.values(), key=lambda x: x["created_at"], reverse=True)
        ]

    def search(self, query: str) -> list[dict]:
        q = query.lower()
        results = []
        for note in self._notes.values():
            if (
                q in note["title"].lower()
                or q in note["body"].lower()
                or any(q in tag.lower() for tag in note["tags"])
            ):
                results.append({
                    "id": note["id"],
                    "title": note["title"],
                    "snippet": note["body"][:120] + ("…" if len(note["body"]) > 120 else ""),
                    "tags": note["tags"],
                    "created_at": note["created_at"],
                })
        return results


_store = NoteStore()  # module-level singleton

# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Create a new note with a title, a markdown body, and optional tags. "
        "Returns the created note with its unique ID."
    )
)
def create_note(
    title: Annotated[str, "Short, descriptive title for the note"],
    body: Annotated[str, "Markdown-formatted note body / content"],
    tags: Annotated[list[str], "Optional list of tag strings for categorisation"] = None,
) -> dict:
    """Create and persist a new note. Returns the full note dict."""
    if not title.strip():
        return {"error": "Title must not be empty."}
    if len(body) > 50_000:
        return {"error": "Note body too large (max 50 000 chars)."}
    return _store.create(title, body, tags)


@mcp.tool(
    description="Retrieve a single note by its ID. Returns the full note including body."
)
def get_note(
    note_id: Annotated[str, "8-character note ID returned by create_note"],
) -> dict:
    """Fetch note *note_id*. Returns note dict or error if not found."""
    note = _store.get(note_id)
    if note is None:
        return {"error": f"Note '{note_id}' not found."}
    return note


@mcp.tool(
    description=(
        "List all stored notes (ID, title, tags, created_at). "
        "Does not include full note bodies to keep the response compact."
    )
)
def list_notes() -> dict:
    """Return a summary list of all notes."""
    notes = _store.list_all()
    return {"count": len(notes), "notes": notes}


@mcp.tool(
    description=(
        "Search notes by keyword. Matches against title, body, and tags. "
        "Returns matching notes with a short snippet of the body."
    )
)
def search_notes(
    query: Annotated[str, "Search keyword or phrase"],
) -> dict:
    """Full-text search across all notes."""
    if not query.strip():
        return {"error": "Query must not be empty.", "results": []}
    results = _store.search(query)
    return {"query": query, "count": len(results), "results": results}


@mcp.tool(
    description="Delete a note by ID. Returns success or an error if not found."
)
def delete_note(
    note_id: Annotated[str, "8-character note ID to delete"],
) -> dict:
    """Remove note *note_id* from the store."""
    if _store.delete(note_id):
        logger.info("Note deleted: id=%s", note_id)
        return {"deleted": True, "note_id": note_id}
    return {"deleted": False, "error": f"Note '{note_id}' not found."}


@mcp.tool(
    description=(
        "Update an existing note's title and/or body. "
        "Pass only the fields you want to change."
    )
)
def update_note(
    note_id: Annotated[str, "8-character note ID to update"],
    title: Annotated[str | None, "New title (omit to keep current)"] = None,
    body: Annotated[str | None, "New body (omit to keep current)"] = None,
) -> dict:
    """Edit *note_id* fields in-place."""
    if title is None and body is None:
        return {"error": "Provide at least one of title or body to update."}
    updated = _store.update(note_id, title, body)
    if updated is None:
        return {"error": f"Note '{note_id}' not found."}
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Notes MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    logger.info("Starting Notes MCP server (transport=%s)", args.transport)
    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")
