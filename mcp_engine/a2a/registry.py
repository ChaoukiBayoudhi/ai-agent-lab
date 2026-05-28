"""
Agent Registry
==============
A central directory of all registered agents and their capabilities.

Why a registry?
---------------
In a multi-agent system, the coordinator needs to know *who can do what*.
The registry decouples agents from each other — agents don't hard-code
references to other agents; they ask the registry.

Pattern: Service Locator / Directory
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .schemas import AgentCard

if TYPE_CHECKING:
    from ..agents.base_agent import BaseAgent

logger = logging.getLogger("mcp_engine.registry")


class AgentRegistry:
    """
    Singleton registry that maps agent names → (AgentCard, BaseAgent instance).

    Usage::

        registry = AgentRegistry()
        registry.register(agent_instance, card)
        agent = registry.get_agent("calculator_agent")
        agents = registry.find_by_capability("math")
    """

    _instance: "AgentRegistry | None" = None

    def __new__(cls) -> "AgentRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents: dict[str, "BaseAgent"] = {}
            cls._instance._cards: dict[str, AgentCard] = {}
        return cls._instance

    def register(self, agent: "BaseAgent", card: AgentCard) -> None:
        """Add an agent to the registry."""
        self._agents[card.name] = agent
        self._cards[card.name] = card
        logger.info("Registered agent: %s (%s)", card.name, card.capabilities)

    def get_agent(self, name: str) -> "BaseAgent | None":
        return self._agents.get(name)

    def get_card(self, name: str) -> AgentCard | None:
        return self._cards.get(name)

    def find_by_capability(self, capability: str) -> list[str]:
        """Return names of agents that have *capability*."""
        return [
            name
            for name, card in self._cards.items()
            if capability in card.capabilities
        ]

    def all_cards(self) -> list[dict]:
        return [c.to_dict() for c in self._cards.values()]

    def all_names(self) -> list[str]:
        return list(self._agents.keys())

    def reset(self) -> None:
        """Clear all registrations (useful in tests)."""
        self._agents.clear()
        self._cards.clear()
