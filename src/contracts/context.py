from typing import Annotated

from pydantic import AwareDatetime, Field

from src.contracts.base import Contract
from src.contracts.enums import Environment
from src.contracts.tasks import AgentResult


class EvidenceReference(Contract):
    evidence_id: str
    incident_id: str
    source: str
    object_uri: str
    checksum_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    observed_at: AwareDatetime


class IncidentContext(Contract):
    incident_id: str
    version: Annotated[int, Field(ge=1)]
    environment: Environment
    status: str
    agent_results: tuple[AgentResult, ...] = ()
    evidence_refs: tuple[EvidenceReference, ...] = ()
