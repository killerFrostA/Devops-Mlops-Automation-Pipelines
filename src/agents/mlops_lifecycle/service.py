from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import ValidationError

from src.agents.base import SkeletonAgent
from src.agents.mlops_lifecycle.health import InsufficientSamplesError, RuleBasedModelHealthAssessor
from src.agents.mlops_lifecycle.ports import (
    BaselineRepository,
    LabelRepository,
    ModelHealthAssessor,
    PredictionRepository,
)
from src.agents.mlops_lifecycle.window_builder import build_model_health_input
from src.contracts.enums import AgentId, ResultStatus
from src.contracts.mlops import ModelHealthInput, ModelHealthReport
from src.contracts.mlops_observations import ModelWindowQuery, WindowBuildRequest
from src.contracts.tasks import AgentResult, AgentTask
from src.platform.errors import IntegrationNotConfiguredError


class MLOpsLifecycleAgent(SkeletonAgent):
    agent_id = AgentId.MLOPS_LIFECYCLE
    responsibility = "Assess model health, evaluate candidates and recommend controlled promotion"
    implementation_status: Literal["NOT_IMPLEMENTED", "PARTIAL"] = "PARTIAL"

    def __init__(
        self,
        assessor: ModelHealthAssessor | None = None,
        *,
        baselines: BaselineRepository | None = None,
        predictions: PredictionRepository | None = None,
        labels: LabelRepository | None = None,
    ) -> None:
        self._assessor = assessor or RuleBasedModelHealthAssessor()
        self._baselines = baselines
        self._predictions = predictions
        self._labels = labels

    async def assess_model_health(self, request: ModelHealthInput) -> ModelHealthReport:
        return await self._assessor.assess(request)

    async def assess_observation_window(self, request: WindowBuildRequest) -> ModelHealthReport:
        return await self.assess_model_health(build_model_health_input(request))

    async def assess_model_window(self, query: ModelWindowQuery) -> ModelHealthReport:
        if self._baselines is None or self._predictions is None or self._labels is None:
            raise IntegrationNotConfiguredError("Model-health data readers are not configured")
        baseline = await self._baselines.get(query.model_name, query.model_version)
        prediction_batch = await self._predictions.list_window(
            query.model_name, query.model_version, query.window_start, query.window_end
        )
        if not prediction_batch.records:
            raise ValueError("No predictions in the requested model window")
        label_batch = await self._labels.list_confirmed(
            tuple(record.observation_id for record in prediction_batch.records), query.as_of
        )
        window = WindowBuildRequest(
            baseline=baseline,
            window_start=query.window_start,
            window_end=query.window_end,
            as_of=query.as_of,
            predictions=prediction_batch.records,
            labels=label_batch.records,
            predictions_evidence_ref=prediction_batch.evidence_ref,
            labels_evidence_ref=label_batch.evidence_ref,
            dataset_ref=query.dataset_ref,
        )
        return await self.assess_observation_window(window)

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
