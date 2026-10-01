"""Deterministic incident supervisor with fail-closed agent dispatch."""

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from uuid import NAMESPACE_URL, uuid5

from pydantic import JsonValue

from src.agents.base import Agent
from src.contracts.enums import AgentId
from src.contracts.events import EventEnvelope
from src.contracts.payloads import (
    AnomalyDetected,
    DiagnosisCompleted,
    ModelHealthAssessed,
    ReleaseAssessment,
    ResourceRecommendation,
)
from src.contracts.tasks import AgentResult, AgentTask, TaskContext
from src.platform.orchestrator.decision.contracts import (
    DecisionDisposition,
    DecisionPlan,
)
from src.platform.orchestrator.decision.engine import DecisionEngine
from src.platform.orchestrator.repository import WorkflowConflictError, WorkflowRepository
from src.platform.orchestrator.state import (
    EventReceipt,
    TaskRecord,
    TaskStatus,
    WorkflowStage,
    WorkflowState,
)
from src.platform.ports import ContextReader

logger = logging.getLogger(__name__)


class WorkflowError(RuntimeError):
    """An event, task or transition violates the workflow boundary."""


class WorkflowNotFoundError(WorkflowError):
    """The requested incident has no checkpoint."""


OBJECTIVES: dict[AgentId, str] = {
    AgentId.DIAGNOSIS: "Assess probable causes using bounded incident evidence",
    AgentId.SECURITY_QUALITY: "Assess release and security quality without side effects",
    AgentId.RESOURCE_OPTIMIZATION: "Assess bounded resource options without side effects",
    AgentId.MLOPS_LIFECYCLE: "Assess model health using registered evidence without side effects",
}
STAGES: dict[str, WorkflowStage] = {
    "monitoring.anomaly.detected": WorkflowStage.DIAGNOSING,
    "diagnosis.completed": WorkflowStage.ASSESSING,
    "resource.recommendation.created": WorkflowStage.ASSESSING,
    "mlops.model.health.assessed": WorkflowStage.ASSESSING,
    "quality.release.assessed": WorkflowStage.ASSESSING,
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _fingerprint(event: EventEnvelope) -> str:
    serialized = json.dumps(
        event.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validated_state(state: WorkflowState, **changes: object) -> WorkflowState:
    document = state.model_dump(mode="json")
    document.update(changes)
    return WorkflowState.model_validate(document)


def _decision_stage(plan: DecisionPlan) -> WorkflowStage:
    return {
        DecisionDisposition.ROUTE_ANALYSIS: STAGES[plan.topic],
        DecisionDisposition.WAIT_FOR_EVIDENCE: WorkflowStage.WAITING_FOR_EVIDENCE,
        DecisionDisposition.REVIEW_REQUIRED: WorkflowStage.REVIEW_REQUIRED,
        DecisionDisposition.BLOCKED: WorkflowStage.BLOCKED,
        DecisionDisposition.NO_ACTION: WorkflowStage.WAITING_FOR_EVENT,
    }[plan.disposition]


class Orchestrator:
    """Coordinate analysis only; all action, policy and HITL paths remain closed.

    The repository is a checkpoint, not the source of operational facts. ContextReader
    supplies the authoritative incident snapshot. A claimed task is never replayed
    automatically after process loss; it must be reconciled by the future worker layer.
    """

    def __init__(
        self,
        repository: WorkflowRepository,
        context_reader: ContextReader,
        agents: Mapping[AgentId, Agent],
        *,
        clock: Callable[[], datetime] = _utc_now,
        task_timeout: timedelta = timedelta(minutes=5),
        max_event_age: timedelta = timedelta(minutes=15),
        decision_engine: DecisionEngine | None = None,
    ) -> None:
        if task_timeout <= timedelta(0) or max_event_age <= timedelta(0):
            raise ValueError("Timeout and maximum event age must be positive")
        for agent_id, agent in agents.items():
            if agent.agent_id != agent_id:
                raise ValueError("Agent registry key does not match agent identity")
        self._repository = repository
        self._context_reader = context_reader
        self._agents = dict(agents)
        self._clock = clock
        self._task_timeout = task_timeout
        self._max_event_age = max_event_age
        self._decision_engine = decision_engine or DecisionEngine()

    def get(self, incident_id: str) -> WorkflowState:
        state = self._repository.get(incident_id)
        if state is None:
            raise WorkflowNotFoundError("No workflow checkpoint for this incident")
        return state

    async def ingest(self, incoming: EventEnvelope) -> WorkflowState:
        """Plan a validated event exactly once, never dispatch during this transaction."""
        event = EventEnvelope.model_validate(incoming.model_dump(mode="json"))
        if event.target != "orchestrator":
            raise WorkflowError("Event is not addressed to the orchestrator")
        if len(event.model_dump_json().encode("utf-8")) > 65_536:
            raise WorkflowError("Event exceeds the bounded orchestration envelope")
        if len(event.evidence_refs) > 64 or len(set(event.evidence_refs)) != len(
            event.evidence_refs
        ):
            raise WorkflowError("Invalid or excessive evidence references")
        fingerprint = _fingerprint(event)
        if event.topic == "monitoring.anomaly.detected":
            anomaly = AnomalyDetected.model_validate(event.payload)
            model_id = anomaly.signals.get("model_id")
            model_version = anomaly.signals.get("model_version")
            if (model_id is None) != (model_version is None):
                raise WorkflowError("Model-related anomaly has incomplete model identity")
            if model_id is not None and (
                not isinstance(model_id, str)
                or not model_id.strip()
                or not isinstance(model_version, str)
                or not model_version.strip()
            ):
                raise WorkflowError("Model identity must contain non-empty strings")
        if event.topic not in STAGES:
            raise WorkflowError("Event has no supported analysis route")

        decision: DecisionPlan | None = None
        for _ in range(16):
            prior = self._repository.get(event.correlation_id)
            if prior is not None:
                match = next(
                    (
                        receipt
                        for receipt in prior.receipts
                        if receipt.message_id == event.message_id
                        or receipt.idempotency_key == event.idempotency_key
                    ),
                    None,
                )
                if match is not None:
                    if (
                        match.message_id == event.message_id
                        and match.idempotency_key == event.idempotency_key
                        and match.fingerprint_sha256 == fingerprint
                    ):
                        return prior
                    raise WorkflowError("Event identity or idempotency key was reused")
                if prior.stage != WorkflowStage.WAITING_FOR_EVENT:
                    raise WorkflowError("Previous analysis has not completed successfully")
                if (
                    prior.environment != event.environment
                    or prior.service_id != event.service.service_id
                    or prior.namespace != event.service.namespace
                    or prior.service_version != event.service.version
                    or prior.trace_id != event.trace_id
                ):
                    raise WorkflowError("Event scope differs from the existing incident")
                if len(prior.receipts) >= 128:
                    raise WorkflowError("Workflow checkpoint limit reached")
            elif event.topic != "monitoring.anomaly.detected":
                raise WorkflowError("An incident must begin with an anomaly event")

            now = self._clock()
            if now.tzinfo is None:
                raise ValueError("Orchestrator clock must be timezone-aware")
            if event.timestamp > now + timedelta(minutes=5):
                raise WorkflowError("Event timestamp is too far in the future")
            if now - event.timestamp > self._max_event_age:
                raise WorkflowError("Event is too old for automatic planning")
            deadline = min(event.deadline or now + self._task_timeout, now + self._task_timeout)
            if deadline <= now:
                raise WorkflowError("Event deadline has expired")

            context = await self._context_reader.get_incident_context(event.correlation_id)
            if (
                context.incident_id != event.correlation_id
                or context.environment != event.environment
            ):
                raise WorkflowError("Authoritative incident context has a different scope")
            if any(
                evidence.incident_id != context.incident_id for evidence in context.evidence_refs
            ):
                raise WorkflowError("Authoritative context contains cross-incident evidence")
            trusted_ids = {evidence.evidence_id for evidence in context.evidence_refs}
            cited_ids = set(event.evidence_refs)
            if event.topic == "diagnosis.completed":
                diagnosis = DiagnosisCompleted.model_validate(event.payload)
                cited_ids.update(
                    reference
                    for hypothesis in diagnosis.hypotheses
                    for reference in hypothesis.evidence_refs
                )
                for intent in diagnosis.recommended_actions:
                    if (
                        intent.incident_id != event.correlation_id
                        or intent.environment != event.environment
                        or intent.target.service != event.service.service_id
                        or intent.target.namespace != event.service.namespace
                    ):
                        raise WorkflowError("Candidate action has a different incident scope")
            elif event.topic == "resource.recommendation.created":
                resource = ResourceRecommendation.model_validate(event.payload)
                intent = resource.candidate_action
                if (
                    intent.incident_id != event.correlation_id
                    or intent.environment != event.environment
                    or intent.target.service != event.service.service_id
                    or intent.target.namespace != event.service.namespace
                ):
                    raise WorkflowError("Candidate action has a different incident scope")
            elif event.topic == "mlops.model.health.assessed":
                model_health = ModelHealthAssessed.model_validate(event.payload)
                cited_ids.update(model_health.evidence_refs)
            elif event.topic == "quality.release.assessed":
                quality = ReleaseAssessment.model_validate(event.payload)
                cited_ids.update(quality.evidence_refs)
            if len(cited_ids) > 64 or not cited_ids.issubset(trusted_ids):
                raise WorkflowError("Event cites excessive or unregistered evidence")
            if decision is None:
                decision = await self._decision_engine.decide(
                    event, history=prior.decisions if prior is not None else ()
                )
            targets = decision.next_agents
            if prior is not None and len(prior.tasks) + len(targets) > 256:
                raise WorkflowError("Workflow checkpoint limit reached")
            stage = _decision_stage(decision)

            receipt = EventReceipt(
                message_id=event.message_id,
                idempotency_key=event.idempotency_key,
                fingerprint_sha256=fingerprint,
            )
            signals = self._bounded_signals(event) if targets else {}
            new_tasks = tuple(
                TaskRecord(
                    task=AgentTask(
                        task_id=str(uuid5(NAMESPACE_URL, f"{event.message_id}:{agent_id}")),
                        incident_id=event.correlation_id,
                        agent_id=agent_id,
                        objective=OBJECTIVES[agent_id],
                        deadline=deadline,
                        task_context=TaskContext(
                            service_id=event.service.service_id,
                            namespace=event.service.namespace,
                            environment=event.environment,
                            signals=signals,
                            evidence_refs=tuple(sorted(cited_ids)),
                            constraints=("analysis_only", "no_side_effects"),
                        ),
                    )
                )
                for agent_id in targets
            )
            if prior is None:
                state = WorkflowState(
                    incident_id=event.correlation_id,
                    stage=stage,
                    environment=event.environment,
                    service_id=event.service.service_id,
                    namespace=event.service.namespace,
                    service_version=event.service.version,
                    priority=event.priority,
                    severity=event.severity,
                    trace_id=event.trace_id,
                    receipts=(receipt,),
                    decisions=(decision,),
                    tasks=new_tasks,
                    updated_at=now,
                )
                try:
                    self._repository.create(state)
                    return state
                except WorkflowConflictError:
                    continue
            state = _validated_state(
                prior,
                revision=prior.revision + 1,
                stage=stage,
                priority=event.priority,
                severity=event.severity,
                receipts=(*prior.receipts, receipt),
                decisions=(*prior.decisions, decision),
                tasks=(*prior.tasks, *new_tasks),
                updated_at=now,
            )
            try:
                self._repository.replace(state, expected_revision=prior.revision)
                return state
            except WorkflowConflictError:
                continue
        raise WorkflowConflictError("Too many concurrent workflow updates")

    @staticmethod
    def _bounded_signals(event: EventEnvelope) -> dict[str, JsonValue]:
        if event.topic == "monitoring.anomaly.detected":
            anomaly = AnomalyDetected.model_validate(event.payload)
            signals: dict[str, JsonValue] = {
                "service_version": event.service.version,
                "anomaly_id": anomaly.anomaly_id,
                "detector": anomaly.detector,
                "anomaly_score": anomaly.anomaly_score,
            }
            if "model_id" in anomaly.signals:
                signals["model_id"] = anomaly.signals["model_id"]
                signals["model_version"] = anomaly.signals["model_version"]
            return signals
        if event.topic == "diagnosis.completed":
            diagnosis = DiagnosisCompleted.model_validate(event.payload)
            return {
                "service_version": event.service.version,
                "diagnosis_id": diagnosis.diagnosis_id,
                "primary_root_cause": diagnosis.primary_root_cause,
                "candidate_action_ids": [
                    action.action_id for action in diagnosis.recommended_actions
                ],
                "requires_additional_evidence": diagnosis.requires_additional_evidence,
            }
        if event.topic == "resource.recommendation.created":
            recommendation = ResourceRecommendation.model_validate(event.payload)
            return {
                "service_version": event.service.version,
                "recommendation_id": recommendation.recommendation_id,
                "candidate_action_id": recommendation.candidate_action.action_id,
                "candidate_action_type": recommendation.candidate_action.action.value,
                "confidence": recommendation.confidence,
            }
        if event.topic == "mlops.model.health.assessed":
            assessment = ModelHealthAssessed.model_validate(event.payload)
            model_signals: dict[str, JsonValue] = {
                "service_version": event.service.version,
                "assessment_id": assessment.assessment_id,
                "model_name": assessment.model_name,
                "model_version": assessment.model_version,
                "drift_score": assessment.drift_score,
                "decision": assessment.decision,
            }
            if assessment.dataset_ref is not None:
                model_signals["dataset_ref"] = assessment.dataset_ref
            return model_signals
        raise WorkflowError("No bounded task context for this decision")

    def _claim(self, incident_id: str, task_id: str) -> AgentTask | None:
        for _ in range(16):
            state = self.get(incident_id)
            record = next((item for item in state.tasks if item.task.task_id == task_id), None)
            if record is None:
                raise WorkflowError("Task is absent from workflow")
            if record.status != TaskStatus.PENDING:
                return None
            if state.stage == WorkflowStage.BLOCKED:
                return None
            tasks = tuple(
                TaskRecord(task=item.task, status=TaskStatus.DISPATCHING)
                if item.task.task_id == task_id
                else item
                for item in state.tasks
            )
            updated = _validated_state(
                state, revision=state.revision + 1, tasks=tasks, updated_at=self._clock()
            )
            try:
                self._repository.replace(updated, expected_revision=state.revision)
                return record.task
            except WorkflowConflictError:
                continue
        raise WorkflowConflictError("Too many concurrent dispatch claims")

    def _finish(
        self,
        incident_id: str,
        task_id: str,
        *,
        result: AgentResult | None = None,
        error_code: str | None = None,
        timed_out: bool = False,
    ) -> WorkflowState:
        for _ in range(16):
            state = self.get(incident_id)
            record = next((item for item in state.tasks if item.task.task_id == task_id), None)
            if record is None:
                raise WorkflowError("Result has no planned task")
            if record.status != TaskStatus.DISPATCHING:
                if result is not None and record.result == result:
                    return state
                raise WorkflowError("Task is not awaiting this result")
            if result is not None:
                status = TaskStatus(result.status.value)
                replacement = TaskRecord(task=record.task, status=status, result=result)
            else:
                status = TaskStatus.TIMED_OUT if timed_out else TaskStatus.FAILED
                replacement = TaskRecord(task=record.task, status=status, error_code=error_code)
            tasks = tuple(
                replacement if item.task.task_id == task_id else item for item in state.tasks
            )
            if status != TaskStatus.SUCCEEDED:
                stage = WorkflowStage.BLOCKED
            elif all(item.status == TaskStatus.SUCCEEDED for item in tasks):
                stage = WorkflowStage.WAITING_FOR_EVENT
            else:
                stage = state.stage
            updated = _validated_state(
                state,
                revision=state.revision + 1,
                stage=stage,
                tasks=tasks,
                updated_at=self._clock(),
            )
            try:
                self._repository.replace(updated, expected_revision=state.revision)
                return updated
            except WorkflowConflictError:
                continue
        raise WorkflowConflictError("Too many concurrent result updates")

    async def record_result(self, incoming: AgentResult) -> WorkflowState:
        """Accept one specialist result after checking task identity and evidence."""
        result = AgentResult.model_validate(incoming.model_dump(mode="json"))
        state = self.get(result.incident_id)
        record = next((item for item in state.tasks if item.task.task_id == result.task_id), None)
        if record is None or record.task.agent_id != result.agent_id:
            raise WorkflowError("Agent result does not match a planned task")
        if record.result == result:
            return state
        context = await self._context_reader.get_incident_context(result.incident_id)
        if context.incident_id != state.incident_id or context.environment != state.environment:
            raise WorkflowError("Authoritative context changed incident scope")
        if any(evidence.incident_id != context.incident_id for evidence in context.evidence_refs):
            raise WorkflowError("Authoritative context contains cross-incident evidence")
        trusted_ids = {evidence.evidence_id for evidence in context.evidence_refs}
        if not set(result.evidence_refs).issubset(trusted_ids):
            raise WorkflowError("Agent result cites unregistered evidence")
        return self._finish(result.incident_id, result.task_id, result=result)

    async def _run_one(self, incident_id: str, task_id: str) -> None:
        task = self._claim(incident_id, task_id)
        if task is None:
            return
        remaining = (task.deadline - self._clock()).total_seconds()
        if remaining <= 0:
            self._finish(incident_id, task_id, error_code="DEADLINE_EXPIRED", timed_out=True)
            return
        agent = self._agents.get(task.agent_id)
        if agent is None:
            self._finish(incident_id, task_id, error_code="AGENT_NOT_REGISTERED")
            return
        try:
            raw_result = await asyncio.wait_for(agent.handle(task), timeout=remaining)
        except TimeoutError:
            self._finish(incident_id, task_id, error_code="AGENT_TIMEOUT", timed_out=True)
            return
        except Exception as exc:
            logger.warning("Agent invocation failed: %s", type(exc).__name__)
            self._finish(incident_id, task_id, error_code="AGENT_ERROR")
            return
        try:
            result = AgentResult.model_validate(raw_result.model_dump(mode="json"))
            if (
                result.task_id != task.task_id
                or result.incident_id != task.incident_id
                or result.agent_id != task.agent_id
            ):
                raise WorkflowError("Agent returned a result for a different task")
            await self.record_result(result)
        except (ValueError, AttributeError, WorkflowError) as exc:
            logger.warning("Agent result rejected: %s", type(exc).__name__)
            self._finish(incident_id, task_id, error_code="INVALID_AGENT_RESULT")

    async def dispatch_pending(self, incident_id: str) -> WorkflowState:
        """Dispatch only PENDING tasks. Claimed tasks are not automatically replayed."""
        state = self.get(incident_id)
        pending = [item.task.task_id for item in state.tasks if item.status == TaskStatus.PENDING]
        if state.stage == WorkflowStage.BLOCKED:
            return state
        await asyncio.gather(*(self._run_one(incident_id, task_id) for task_id in pending))
        return self.get(incident_id)

    async def run_event(self, event: EventEnvelope) -> WorkflowState:
        """Local adapter for plan followed by in-process analysis dispatch."""
        state = await self.ingest(event)
        return await self.dispatch_pending(state.incident_id)
