"""
Models
======
Persist calculation history and agent chat sessions so the UI
can display logs and the instructor can inspect student interactions.
"""

from django.db import models
from django.utils import timezone


class CalculationLog(models.Model):
    """Records every direct calculator tool call made via the API."""

    expression = models.TextField(help_text="Raw input expression or question")
    result     = models.TextField(blank=True, help_text="Computed result as JSON string")
    tool_used  = models.CharField(max_length=64, blank=True)
    success    = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Calculation Log"
        verbose_name_plural = "Calculation Logs"

    def __str__(self) -> str:
        return f"[{self.created_at:%Y-%m-%d %H:%M}] {self.expression[:60]}"

    @property
    def result_preview(self) -> str:
        """Short readable preview of the stored JSON result."""
        try:
            import json
            data = json.loads(self.result)
            # Grab the first meaningful value
            for key in ("result", "solutions", "mean", "converted", "error"):
                if key in data:
                    val = data[key]
                    return str(val)[:80]
            return str(data)[:80]
        except Exception:
            return self.result[:80]


class AgentSession(models.Model):
    """Records each complete agent run (user message → final answer)."""

    session_id   = models.CharField(max_length=32, unique=True)
    user_message = models.TextField()
    agent_name   = models.CharField(max_length=64, blank=True)
    answer       = models.TextField(blank=True)
    tool_calls   = models.JSONField(default=list)
    steps        = models.IntegerField(default=0)
    error        = models.TextField(blank=True)
    created_at   = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Agent Session"
        verbose_name_plural = "Agent Sessions"

    def __str__(self) -> str:
        return f"[{self.created_at:%Y-%m-%d %H:%M}] {self.agent_name}: {self.user_message[:60]}"


class A2ATaskLog(models.Model):
    """Logs every A2A task delegation for educational / audit purposes."""

    task_id    = models.CharField(max_length=32)
    from_agent = models.CharField(max_length=64)
    to_agent   = models.CharField(max_length=64)
    message    = models.TextField()
    result     = models.TextField(blank=True)
    state      = models.CharField(max_length=20, default="submitted")
    tool_calls = models.JSONField(default=list)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "A2A Task Log"
        verbose_name_plural = "A2A Task Logs"

    def __str__(self) -> str:
        return f"{self.from_agent} → {self.to_agent}: {self.message[:60]}"
