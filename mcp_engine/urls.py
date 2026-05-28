"""URL configuration for the mcp_engine app."""

from django.urls import path
from . import views

app_name = "mcp_engine"

urlpatterns = [
    # ── HTML pages ─────────────────────────────────────────────────────────
    path("",                    views.index,          name="index"),
    path("calculator/",         views.calculator_ui,  name="calculator"),
    path("agent/",              views.agent_ui,        name="agent"),
    path("a2a/",                views.a2a_ui,          name="a2a"),
    path("notes/",              views.notes_ui,        name="notes"),
    path("history/",            views.history_ui,      name="history"),

    # ── JSON API ────────────────────────────────────────────────────────────
    path("api/calculate/",      views.api_calculate,   name="api_calculate"),
    path("api/agent/chat/",     views.api_agent_chat,  name="api_agent_chat"),
    path("api/a2a/delegate/",   views.api_a2a_task,    name="api_a2a_task"),
    path("api/agents/",         views.api_agents,      name="api_agents"),

    # ── A2A well-known ──────────────────────────────────────────────────────
    path(".well-known/agent.json", views.agent_well_known, name="agent_card"),
]
