from src.contracts.context import EvidenceReference, IncidentContext
from src.contracts.tasks import AgentResult
from src.platform.errors import IntegrationNotConfiguredError


class ContextService:
    """Replace methods with transaction-backed implementations and authenticated ACLs."""

    async def get_incident_context(self, incident_id: str) -> IncidentContext:
        raise IntegrationNotConfiguredError("Context PostgreSQL repository is not implemented")

    async def append_agent_result(self, result: AgentResult) -> str:
        raise IntegrationNotConfiguredError("Context result persistence is not implemented")

    async def append_evidence(self, evidence: EvidenceReference) -> str:
        raise IntegrationNotConfiguredError("Context evidence persistence is not implemented")
