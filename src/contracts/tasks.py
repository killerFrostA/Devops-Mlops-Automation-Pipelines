from typing import Annotated

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from src.contracts.base import Contract
from src.contracts.enums import AgentId, Environment, ResultStatus
from src.contracts.mlops_projects import ProjectEvaluationReport
from src.contracts.payloads import ModelHealthAssessed


class TaskContext(Contract):
    """Bounded disclosure supplied by the orchestrator, never a full context snapshot."""

    service_id: Annotated[str, Field(min_length=1)]
    namespace: Annotated[str, Field(min_length=1)]
    environment: Environment
    signals: dict[str, JsonValue] = Field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()


class AgentTask(Contract):
    task_id: Annotated[str, Field(min_length=1)]
    incident_id: Annotated[str, Field(min_length=1)]
    agent_id: AgentId
    objective: Annotated[str, Field(min_length=1)]
    task_context: TaskContext
    deadline: AwareDatetime


class AgentResult(Contract):
    """Execution status and optional validated specialist domain output."""

    result_id: Annotated[str, Field(min_length=1)]
    task_id: Annotated[str, Field(min_length=1)]
    incident_id: Annotated[str, Field(min_length=1)]
    agent_id: AgentId
    status: ResultStatus
    rationale_summary: str
    evidence_refs: tuple[str, ...] = ()
    confidence: Annotated[float | None, Field(ge=0, le=1, allow_inf_nan=False)] = None
    payload: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_domain_payload(self) -> "AgentResult":
        if self.agent_id is AgentId.MLOPS_LIFECYCLE and self.status is ResultStatus.SUCCEEDED:
            if "task_type" in self.payload:
                ProjectEvaluationReport.model_validate(self.payload)
            else:
                ModelHealthAssessed.model_validate(self.payload)
        return self
