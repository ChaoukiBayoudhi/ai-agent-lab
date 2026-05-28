"""Admin registrations for mcp_engine models."""

from django.contrib import admin
from .models import CalculationLog, AgentSession, A2ATaskLog


@admin.register(CalculationLog)
class CalculationLogAdmin(admin.ModelAdmin):
    list_display   = ("created_at", "tool_used", "success", "expression")
    list_filter    = ("tool_used", "success")
    search_fields  = ("expression", "result")
    readonly_fields = ("created_at",)


@admin.register(AgentSession)
class AgentSessionAdmin(admin.ModelAdmin):
    list_display   = ("created_at", "agent_name", "session_id", "user_message")
    list_filter    = ("agent_name",)
    search_fields  = ("user_message", "answer", "session_id")
    readonly_fields = ("created_at", "session_id")


@admin.register(A2ATaskLog)
class A2ATaskLogAdmin(admin.ModelAdmin):
    list_display   = ("created_at", "from_agent", "to_agent", "state", "message")
    list_filter    = ("state", "from_agent", "to_agent")
    search_fields  = ("message", "result", "task_id")
    readonly_fields = ("created_at",)
