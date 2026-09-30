from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import ValidationError

from src.agents.base import SkeletonAgent
from src.agents.mlops_lifecycle.evaluation.engine import ProjectEvaluationService
from src.agents.mlops_lifecycle.health.assessor import (
    InsufficientSamplesError,
    RuleBasedModelHealthAssessor,
)
from src.agents.mlops_lifecycle.health.window_builder import build_model_health_input
from src.agents.mlops_lifecycle.interfaces.ports import (
    BaselineRepository,
    LabelRepository,
    ModelHealthAssessor,
    ModelRegistry,
    PredictionRepository,
    TrainingPipeline,
)
from src.contracts.enums import AgentId, ResultStatus
from src.contracts.mlops import ModelHealthInput, ModelHealthReport
from src.contracts.mlops_observations import ModelWindowQuery, WindowBuildRequest
from src.contracts.mlops_projects import EvaluationBatch, ProjectEvaluationReport, ProjectManifest
from src.contracts.mlops_training import ModelLifecycleReport, TrainingDataset, TrainingPolicy
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
        training: TrainingPipeline | None = None,
        registry: ModelRegistry | None = None,
        project_evaluation: ProjectEvaluationService | None = None,
    ) -> None:
        self._assessor = assessor or RuleBasedModelHealthAssessor()
        self._baselines = baselines
        self._predictions = predictions
        self._labels = labels
        self._training = training
        self._registry = registry
        self._project_evaluation = project_evaluation or ProjectEvaluationService()

    def register_project(self, manifest: ProjectManifest) -> None:
        """Trusted configuration step; task callers cannot override registered policy."""
        self._project_evaluation.register_project(manifest)

    async def assess_project_batch(self, batch: EvaluationBatch) -> ProjectEvaluationReport:
        return await self._project_evaluation.evaluate(batch)

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

    async def assess_and_train(
        self, query: ModelWindowQuery, dataset: TrainingDataset, policy: TrainingPolicy
    ) -> ModelLifecycleReport:
        health = await self.assess_model_window(query)
        limitations = (
            "This local example uses synthetic history and a fixed-rule champion proxy.",
            "Registration requests review; deployment and production promotion remain external.",
            "The local test metrics and latency do not establish production performance.",
        )
        if health.assessment.decision != "RETRAIN_AND_EVALUATE":
            return ModelLifecycleReport(
                health=health,
                evaluation=None,
                decision="SKIP_TRAINING",
                rationale_summary=(
                    f"Health decision is {health.assessment.decision}; training was skipped"
                ),
                limitations=limitations,
            )
        if dataset.model_name != query.model_name:
            raise ValueError("Training dataset model does not match the assessed model")
        if any(row.observed_at >= query.window_start for row in dataset.observations):
            raise ValueError("Training observations must predate the monitored window")
        if self._training is None or self._registry is None:
            raise IntegrationNotConfiguredError("Training and model registry are not configured")
        evaluation = await self._training.train_and_evaluate(dataset, policy)
        if (
            evaluation.model_name != dataset.model_name
            or evaluation.dataset_ref != dataset.dataset_ref
            or evaluation.policy != policy
        ):
            raise ValueError("Candidate evaluation does not match the requested dataset and policy")
        candidate = evaluation.candidate
        champion = evaluation.champion
        reasons = []
        if candidate.f1 + 1e-12 < champion.f1 + policy.min_f1_gain:
            reasons.append("candidate test F1 gain is below the configured minimum")
        if policy.require_recall_nonregression and candidate.recall + 1e-12 < champion.recall:
            reasons.append("candidate test recall is lower than champion recall")
        if candidate.p95_latency_ms > policy.max_p95_latency_ms:
            reasons.append("candidate p95 inference latency exceeds the limit")
        if reasons:
            return ModelLifecycleReport(
                health=health,
                evaluation=evaluation,
                decision="KEEP_CHAMPION",
                rationale_summary="; ".join(reasons),
                limitations=limitations,
            )
        version = await self._registry.register_candidate(evaluation)
        return ModelLifecycleReport(
            health=health,
            evaluation=evaluation,
            decision="REQUEST_PROMOTION_REVIEW",
            registered_model_version=version,
            rationale_summary=(
                "Candidate passed local comparison; request orchestrator review before promotion"
            ),
            limitations=limitations,
        )

    async def handle(self, task: AgentTask) -> AgentResult:
        if task.agent_id != self.agent_id:
            raise ValueError("Task addressed to a different agent")
        if task.deadline <= datetime.now(UTC):
            raise ValueError("Task deadline has expired")
        if "project_evaluation" in task.task_context.signals:
            try:
                batch = EvaluationBatch.model_validate(
                    task.task_context.signals["project_evaluation"]
                )
            except ValidationError:
                return self._failure(task, "Invalid project_evaluation batch")
            if batch.evidence_ref not in task.task_context.evidence_refs:
                return self._failure(task, "Project evidence must be included in the task context")
            try:
                project_report = await self.assess_project_batch(batch)
            except (ValueError, IntegrationNotConfiguredError, RuntimeError) as error:
                return self._failure(task, str(error))
            if task.deadline <= datetime.now(UTC):
                raise ValueError("Task deadline has expired")
            return AgentResult(
                result_id=str(uuid4()),
                task_id=task.task_id,
                incident_id=task.incident_id,
                agent_id=self.agent_id,
                status=ResultStatus.SUCCEEDED,
                rationale_summary=project_report.rationale_summary,
                evidence_refs=(batch.evidence_ref,),
                payload=project_report.model_dump(mode="json"),
            )
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
