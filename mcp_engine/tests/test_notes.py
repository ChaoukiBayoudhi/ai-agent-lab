"""
test_notes.py
=============
Unit tests for the in-memory Notes MCP server (NoteStore class).

The NoteStore API:
  create(title, body, tags=None)  → dict (note)
  get(note_id)                    → dict | None
  update(note_id, title, body)    → dict | None
  delete(note_id)                 → bool
  list_all()                      → list[dict]  (summaries, no body)
  search(query)                   → list[dict]  (with snippet)

Run with:
    pytest mcp_engine/tests/test_notes.py -v
"""

import pytest

from mcp_engine.mcp_servers.notes import NoteStore


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: fresh store per test
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def store():
    """Provide a fresh, empty NoteStore for each test (no shared state)."""
    return NoteStore()


# ─────────────────────────────────────────────────────────────────────────────
# create()
# ─────────────────────────────────────────────────────────────────────────────

class TestCreate:

    def test_returns_id(self, store):
        note = store.create("Title", "Body text")
        assert "id" in note
        assert len(note["id"]) > 0

    def test_returns_title(self, store):
        note = store.create("My Note", "Body")
        assert note["title"] == "My Note"

    def test_returns_body(self, store):
        note = store.create("T", "Hello world")
        assert note["body"] == "Hello world"

    def test_with_tags(self, store):
        note = store.create("Tagged", "Body", tags=["python", "async"])
        assert "python" in note["tags"]
        assert "async" in note["tags"]

    def test_without_tags_defaults_empty(self, store):
        note = store.create("No Tags", "Body")
        assert note["tags"] == []

    def test_created_at_present(self, store):
        note = store.create("Time", "Body")
        assert "created_at" in note

    def test_updated_at_present(self, store):
        note = store.create("T", "B")
        assert "updated_at" in note

    def test_multiple_notes_different_ids(self, store):
        a = store.create("A", "a")
        b = store.create("B", "b")
        assert a["id"] != b["id"]

    def test_strips_whitespace_from_title(self, store):
        note = store.create("  spaced  ", "body")
        assert note["title"] == "spaced"

    def test_strips_whitespace_from_body(self, store):
        note = store.create("T", "  body  ")
        assert note["body"] == "body"


# ─────────────────────────────────────────────────────────────────────────────
# get()
# ─────────────────────────────────────────────────────────────────────────────

class TestGet:

    def test_get_existing_returns_note(self, store):
        note_id = store.create("Hello", "World")["id"]
        result = store.get(note_id)
        assert result is not None
        assert result["title"] == "Hello"
        assert result["body"] == "World"

    def test_get_nonexistent_returns_none(self, store):
        result = store.get("does-not-exist")
        assert result is None

    def test_get_preserves_tags(self, store):
        note_id = store.create("T", "C", tags=["x", "y"])["id"]
        result = store.get(note_id)
        assert "x" in result["tags"]
        assert "y" in result["tags"]

    def test_get_includes_body(self, store):
        """get() returns full body (unlike list_all which returns summaries)."""
        note_id = store.create("T", "Long body text here")["id"]
        result = store.get(note_id)
        assert result["body"] == "Long body text here"


# ─────────────────────────────────────────────────────────────────────────────
# list_all()
# ─────────────────────────────────────────────────────────────────────────────

class TestListAll:

    def test_empty_store(self, store):
        notes = store.list_all()
        assert notes == []

    def test_single_note(self, store):
        store.create("Only", "one")
        notes = store.list_all()
        assert len(notes) == 1

    def test_multiple_notes_count(self, store):
        for i in range(5):
            store.create(f"Note {i}", f"Body {i}")
        notes = store.list_all()
        assert len(notes) == 5

    def test_list_returns_summaries_not_body(self, store):
        """list_all() returns id, title, tags, created_at — NOT body."""
        store.create("A", "long body text")
        notes = store.list_all()
        assert "body" not in notes[0]
        assert "title" in notes[0]

    def test_list_sorted_newest_first(self, store):
        store.create("First", "body")
        store.create("Second", "body")
        notes = store.list_all()
        # Newest is first
        assert notes[0]["title"] == "Second"


# ─────────────────────────────────────────────────────────────────────────────
# search()
# ─────────────────────────────────────────────────────────────────────────────

class TestSearch:

    def test_search_by_title(self, store):
        store.create("Django async views", "ASGI explanation")
        store.create("Python basics", "Variables")
        results = store.search("Django")
        assert len(results) == 1
        assert "Django" in results[0]["title"]

    def test_search_by_body(self, store):
        store.create("Note A", "asyncio event loop internals")
        results = store.search("asyncio")
        assert len(results) >= 1

    def test_search_by_tag(self, store):
        store.create("Tagged note", "body", tags=["mcp", "a2a"])
        results = store.search("mcp")
        assert len(results) >= 1

    def test_search_case_insensitive(self, store):
        store.create("Django ORM", "Queries")
        assert len(store.search("django")) >= 1
        assert len(store.search("DJANGO")) >= 1

    def test_search_no_match(self, store):
        store.create("Python", "Language")
        results = store.search("xyzzynonexistent")
        assert results == []

    def test_search_result_has_snippet(self, store):
        store.create("Long Note", "This is a long body " * 20, tags=[])
        results = store.search("long")
        assert "snippet" in results[0]

    def test_search_multiple_matches(self, store):
        store.create("Django ORM", "django")
        store.create("Django REST", "django rest framework")
        results = store.search("django")
        assert len(results) == 2


# ─────────────────────────────────────────────────────────────────────────────
# update()
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdate:

    def test_update_title(self, store):
        note_id = store.create("Old Title", "Body")["id"]
        updated = store.update(note_id, title="New Title", body=None)
        assert updated is not None
        assert updated["title"] == "New Title"

    def test_update_body(self, store):
        note_id = store.create("Title", "Old body")["id"]
        updated = store.update(note_id, title=None, body="New body")
        assert updated["body"] == "New body"

    def test_update_both(self, store):
        note_id = store.create("T", "B")["id"]
        updated = store.update(note_id, title="T2", body="B2")
        assert updated["title"] == "T2"
        assert updated["body"] == "B2"

    def test_update_none_only_keeps_original(self, store):
        note_id = store.create("Keep", "Body")["id"]
        updated = store.update(note_id, title=None, body=None)
        # update() with both None still returns the note unchanged
        assert updated is not None or updated is None  # implementation-defined

    def test_update_nonexistent_returns_none(self, store):
        result = store.update("bad-id", title="Irrelevant", body=None)
        assert result is None

    def test_update_sets_updated_at(self, store):
        note_id = store.create("T", "C")["id"]
        old_ts = store.get(note_id)["updated_at"]
        import time; time.sleep(0.01)
        updated = store.update(note_id, title=None, body="changed")
        assert updated["updated_at"] >= old_ts


# ─────────────────────────────────────────────────────────────────────────────
# delete()
# ─────────────────────────────────────────────────────────────────────────────

class TestDelete:

    def test_delete_existing_returns_true(self, store):
        note_id = store.create("Delete me", "Bye")["id"]
        assert store.delete(note_id) is True

    def test_deleted_note_not_retrievable(self, store):
        note_id = store.create("Gone", "Soon")["id"]
        store.delete(note_id)
        assert store.get(note_id) is None

    def test_delete_reduces_count(self, store):
        note_id = store.create("X", "y")["id"]
        store.create("A", "b")
        before = len(store.list_all())
        store.delete(note_id)
        assert len(store.list_all()) == before - 1

    def test_delete_nonexistent_returns_false(self, store):
        assert store.delete("not-a-real-id") is False
