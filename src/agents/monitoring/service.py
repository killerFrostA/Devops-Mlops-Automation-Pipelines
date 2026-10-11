"""Agent 1 as seen by the orchestrator: bounded tasks in, AgentResult out.

Two objectives:
- `detect_anomaly`: watch the task's service once (every detector), report what is abnormal
  as an `AnomalyDetected` (also published on `monitoring.anomaly.detected`);
- `verify_recovery`: independently check, over an observation window, whether the action
  named in `task_context.signals["action_id"]` really fixed the service (RecoveryVerified,
  also published on `monitoring.recovery.verified`).

Built without its dependencies (as the shared registry does), the agent keeps the skeleton
behaviour and honestly answers NOT_IMPLEMENTED.
"""

from datetime import UTC, datetime
from uuid import uuid4

from src.agents.base import SkeletonAgent
from src.agents.monitoring.config import WatchedService
from src.agents.monitoring.events import (
    build_detection_event,
    build_recovery_event,
    json_safe,
    new_id,
)
from src.agents.monitoring.grouping import Symptom
from src.agents.monitoring.loop import MonitoringLoop
from src.agents.monitoring.ports import AnomalyDetector, RecoveryVerifier
from src.agents.monitoring.reasoning import noisy_or
from src.contracts.enums import AgentId, ResultStatus, Severity
from src.contracts.events import ServiceIdentity
from src.contracts.payloads import AnomalyDetected
from src.contracts.tasks import AgentResult, AgentTask, TaskContext
from src.platform.ports import EventPublisher

DETECT = "detect_anomaly"
VERIFY = "verify_recovery"

# How much one symptom alone says the service is abnormal, by its severity.
SEVERITY_SCORE = {
    Severity.LOW: 0.3,
    Severity.MEDIUM: 0.6,
    Severity.HIGH: 0.85,
    Severity.CRITICAL: 0.95,
}


class OneShotDetector:
    """AnomalyDetector port: one detection pass on one service, with the loop's detectors."""

    def __init__(self, loop: MonitoringLoop) -> None:
        self._loop = loop
        self.last_severity: Severity = Severity.MEDIUM

    async def detect(self, context: TaskContext) -> AnomalyDetected | None:
        configured = {(s.service_id, s.namespace): s for s in self._loop.config.services}
        service = configured.get(
            (context.service_id, context.namespace),
            WatchedService(service_id=context.service_id, namespace=context.namespace),
        )
        report, symptoms = await self._loop.watch(service)
        if not symptoms:
            return None
        order = list(Severity)
        self.last_severity = max((s.severity for s in symptoms), key=order.index)
        return AnomalyDetected(
            anomaly_id=new_id("ANM"),
            detector="+".join(sorted({s.source for s in symptoms})),
            anomaly_score=round(_score(symptoms), 4),
            signals={
                key: json_safe(value)
                for key, value in {
                    "categories": sorted({s.category.value for s in symptoms}),
                    "symptoms": [
                        {
                            "category": s.category.value,
                            "source": s.source,
                            "severity": s.severity.value,
                            "facts": s.facts,
                        }
                        for s in symptoms
                    ],
                    "detector_errors": dict(report.errors),
                }.items()
            },
        )


def _score(symptoms: list[Symptom]) -> float:
    """Independent sources agreeing on the service raise the score (noisy-OR)."""
    best: dict[tuple[str, str], float] = {}
    for s in symptoms:
        key = (s.category.value, s.source)
        best[key] = max(best.get(key, 0.0), SEVERITY_SCORE[s.severity])
    return noisy_or(list(best.values()))


class MonitoringAgent(SkeletonAgent):
    agent_id = AgentId.MONITORING
    responsibility = "Collect telemetry, detect anomalies and verify recovery"

    def __init__(
        self,
        detector: AnomalyDetector | None = None,
        verifier: RecoveryVerifier | None = None,
        publisher: EventPublisher | None = None,
        versions: dict[str, str] | None = None,
    ) -> None:
        self._detector = detector
        self._verifier = verifier
        self._publisher = publisher
        self._versions = versions or {}

    async def handle(self, task: AgentTask) -> AgentResult:
        if self._detector is None and self._verifier is None:
            return await super().handle(task)  # not wired: honest NOT_IMPLEMENTED
        if task.agent_id != self.agent_id:
            raise ValueError("Task addressed to a different agent")
        if task.deadline <= datetime.now(UTC):
            raise ValueError("Task deadline has expired")
        if task.objective == DETECT and self._detector is not None:
            return await self._detect(task)
        if task.objective == VERIFY and self._verifier is not None:
            return await self._verify(task)
        return self._result(
            task, ResultStatus.FAILED, f"Objectif non pris en charge : {task.objective}"
        )

    async def _detect(self, task: AgentTask) -> AgentResult:
        assert self._detector is not None
        context = task.task_context
        anomaly = await self._detector.detect(context)
        if anomaly is None:
            return self._result(
                task, ResultStatus.SUCCEEDED, f"Aucune anomalie sur {context.service_id}"
            )
        severity = getattr(self._detector, "last_severity", Severity.MEDIUM)
        if self._publisher is not None:
            await self._publisher.publish(
                build_detection_event(
                    anomaly,
                    self._identity(context),
                    context.environment,
                    task.incident_id,
                    severity,
                )
            )
        categories = anomaly.signals.get("categories", [])
        return self._result(
            task,
            ResultStatus.SUCCEEDED,
            f"Anomalie sur {context.service_id} : {categories} (sources {anomaly.detector})",
            evidence=(anomaly.anomaly_id,),
            confidence=anomaly.anomaly_score,
        )

    async def _verify(self, task: AgentTask) -> AgentResult:
        assert self._verifier is not None
        context = task.task_context
        action_id = context.signals.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            return self._result(
                task, ResultStatus.FAILED, "verify_recovery demande signals.action_id"
            )
        result = await self._verifier.verify(action_id, context)
        if self._publisher is not None:
            await self._publisher.publish(
                build_recovery_event(
                    result, self._identity(context), context.environment, task.incident_id
                )
            )
        checks = ", ".join(f"{name}={value}" for name, value in sorted(result.checks.items()))
        return self._result(
            task,
            ResultStatus.SUCCEEDED,
            f"{result.status} apres {action_id} sur {result.window} ({checks})",
            evidence=(result.verification_id,),
        )

    def _identity(self, context: TaskContext) -> ServiceIdentity:
        return ServiceIdentity(
            service_id=context.service_id,
            namespace=context.namespace,
            version=self._versions.get(context.service_id, "unknown"),
        )

    def _result(
        self,
        task: AgentTask,
        status: ResultStatus,
        summary: str,
        evidence: tuple[str, ...] = (),
        confidence: float | None = None,
    ) -> AgentResult:
        return AgentResult(
            result_id=str(uuid4()),
            task_id=task.task_id,
            incident_id=task.incident_id,
            agent_id=self.agent_id,
            status=status,
            rationale_summary=summary,
            evidence_refs=evidence,
            confidence=confidence,
        )
