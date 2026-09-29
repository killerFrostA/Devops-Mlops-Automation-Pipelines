from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import ValidationError

from src.agents.base import SkeletonAgent
from src.agents.mlops_lifecycle.health import InsufficientSamplesError, RuleBasedModelHealthAssessor
from src.agents.mlops_lifecycle.ports import ModelHealthAssessor
from src.contracts.enums import AgentId, ResultStatus
from src.contracts.mlops import ModelHealthInput, ModelHealthReport
from src.contracts.tasks import AgentResult, AgentTask


class MLOpsLifecycleAgent(SkeletonAgent):
    agent_id = AgentId.MLOPS_LIFECYCLE
    responsibility = "Assess model health, evaluate candidates and recommend controlled promotion"
    implementation_status: Literal["NOT_IMPLEMENTED", "PARTIAL"] = "PARTIAL"

    def __init__(self, assessor: ModelHealthAssessor | None = None) -> None:
        self._assessor = assessor or RuleBasedModelHealthAssessor()

    async def assess_model_health(self, request: ModelHealthInput) -> ModelHealthReport:
        return await self._assessor.assess(request)

    async def handle(self, task: AgentTask) -> AgentResult:
        if task.agent_id != self.agent_id:
            raise ValueError("Task addressed to a different agent")
        if task.deadline <= datetime.now(UTC):
            raise ValueError("Task deadline has expired")
        try:
            request = ModelHealthInput.model_validate(task.task_context.signals.get("model_health"))
        except ValidationError:
            return self._failure(task, "Invalid or missing model_health input")
        if not set(request.evidence_refs).issubset(task.task_context.evidence_refs):
            return self._failure(task, "Model-health evidence must be included in the task context")
        try:
            report = await self.assess_model_health(request)
        except InsufficientSamplesError as error:
            return self._failure(task, str(error))
        if task.deadline <= datetime.now(UTC):
            raise ValueError("Task deadline has expired")
        return AgentResult(
            result_id=str(uuid4()),
            task_id=task.task_id,
            incident_id=task.incident_id,
            agent_id=self.agent_id,
            status=ResultStatus.SUCCEEDED,
            rationale_summary=report.rationale_summary,
            evidence_refs=report.assessment.evidence_refs,
            payload=report.assessment.model_dump(mode="json"),
        )

    def _failure(self, task: AgentTask, reason: str) -> AgentResult:
        return AgentResult(
            result_id=str(uuid4()),
            task_id=task.task_id,
            incident_id=task.incident_id,
            agent_id=self.agent_id,
            status=ResultStatus.FAILED,
            rationale_summary=reason,
        )
