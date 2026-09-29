from enum import StrEnum

from src.contracts.base import Contract
from src.contracts.enums import AgentId


class WorkflowStage(StrEnum):
    INGESTED = "INGESTED"
    DIAGNOSING = "DIAGNOSING"
    ASSESSING = "ASSESSING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    CLOSED = "CLOSED"
    BLOCKED = "BLOCKED"


class WorkflowState(Contract):
    incident_id: str
    stage: WorkflowStage = WorkflowStage.INGESTED
    active_agents: tuple[AgentId, ...] = ()
    selected_action_id: str | None = None
    approval_id: str | None = None
