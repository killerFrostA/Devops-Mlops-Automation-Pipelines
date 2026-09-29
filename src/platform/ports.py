"""Dependency inversion boundary. Identity enforcement belongs to server middleware."""

from typing import Protocol

from src.contracts.context import EvidenceReference, IncidentContext
from src.contracts.events import EventEnvelope
from src.contracts.tasks import AgentResult


class ContextAppender(Protocol):
    """The only context interface injected into agents."""

    async def append_agent_result(self, result: AgentResult) -> str: ...

    async def append_evidence(self, evidence: EvidenceReference) -> str: ...


class ContextReader(Protocol):
    """Injected only into the orchestrator, after workload identity authorization."""

    async def get_incident_context(self, incident_id: str) -> IncidentContext: ...


class EventPublisher(Protocol):
    async def publish(self, event: EventEnvelope) -> None: ...


class IdempotencyStore(Protocol):
    """Implement durable atomic reservations and recovery for interrupted actions."""

    async def reserve(self, key: str, fingerprint: str) -> bool: ...

    async def complete(self, key: str, result: AgentResult) -> None: ...
