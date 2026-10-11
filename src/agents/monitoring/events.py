"""Agent 1 -> the rest of the system: alerts and recovery results as v1 contract events.

An alert decided by the loop becomes an `EventEnvelope` on `monitoring.anomaly.detected` with
an `AnomalyDetected` payload; a recovery check becomes `monitoring.recovery.verified`. Both are
validated by the shared contracts (src/contracts) before leaving the agent.

The payload carries observed facts only (what, where, when, how severe, how sure, measured
values). The root cause is the Diagnosis agent's job and is never stated here.

Transport is a port (`EventPublisher`): a JSON Lines file for now, the team's Kafka publisher
once it exists. Publishing failures are reported, never allowed to stop the agent.
"""

import asyncio
import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, JsonValue

from src.agents.monitoring.decision import DecisionRecord
from src.agents.monitoring.grouping import Incident, most_relevant
from src.agents.monitoring.reasoning import Hypothesis
from src.contracts.enums import AgentId, Environment, Priority, Severity
from src.contracts.events import EventEnvelope, ServiceIdentity
from src.contracts.payloads import AnomalyDetected, RecoveryVerified
from src.platform.ports import EventPublisher

ANOMALY_TOPIC = "monitoring.anomaly.detected"
RECOVERY_TOPIC = "monitoring.recovery.verified"
SOURCE = AgentId.MONITORING.value
# topics.yaml: the orchestrator consumes both topics and routes anomalies to Diagnosis.
TARGET = "orchestrator"

PRIORITY_BY_SEVERITY = {
    Severity.CRITICAL: Priority.P0,
    Severity.HIGH: Priority.P1,
    Severity.MEDIUM: Priority.P2,
    Severity.LOW: Priority.P3,
}


class EventsPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    path: str = "experiments/runs/agent1-events.jsonl"  # JSON Lines file, one event per line
    environment: Environment = Environment.LOCAL


def json_safe(value: object) -> JsonValue:
    """Facts as strict JSON: finite numbers rounded, NaN/Inf dropped to null."""
    if isinstance(value, bool) or value is None or isinstance(value, str | int):
        return value
    if isinstance(value, float):
        return round(value, 4) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [json_safe(v) for v in value]
    return str(value)


def _iso(unix_seconds: float) -> str:
    return datetime.fromtimestamp(unix_seconds, UTC).isoformat()


def build_anomaly_event(
    record: DecisionRecord,
    incidents: list[Incident],
    hypotheses: list[Hypothesis],
    environment: Environment,
    version: str = "unknown",
    now: datetime | None = None,
) -> EventEnvelope:
    """Contract event for an alert. `incidents` are the incidents seen in the cycle."""
    if record.target is None or not incidents:
        raise ValueError("An alert needs a target service and at least one incident")
    own = [i for i in incidents if i.service_id == record.target]
    # The target may be a suspect inferred by reasoning (e.g. a database below a failing API),
    # without an incident of its own: the evidence is then the incidents observed elsewhere.
    evidence = own or incidents
    main = most_relevant(evidence)
    categories = sorted(record.categories or {i.category.value for i in evidence})
    sources = sorted({s for i in evidence for s in i.sources})
    raw: dict[str, object] = {
        "categories": categories,
        "observed_on": sorted({i.service_id for i in evidence}),
        "incidents": [
            {
                "incident_id": i.incident_id,
                "service_id": i.service_id,
                "category": i.category.value,
                "severity": i.severity.value,
                "sources": sorted(i.sources),
                "occurrences": i.occurrences,
                "first_seen": _iso(i.opened_at),
                "last_seen": _iso(i.last_seen_at),
                "facts": i.facts,
            }
            for i in sorted(evidence, key=lambda i: i.incident_id)
        ],
        # Where the problem most probably starts (localisation), not why: no root cause.
        "suspects": [
            {"service_id": h.service_id, "probability": round(h.probability, 4)}
            for h in hypotheses[:3]
        ],
        "decision_reason": record.reason,
    }
    if record.alert_id is not None:
        raw["alert_id"] = record.alert_id
    signals = {key: json_safe(value) for key, value in raw.items()}
    payload = AnomalyDetected(
        anomaly_id=f"ANM-{main.incident_id}-{'+'.join(categories)}",
        detector="+".join(sources),
        anomaly_score=round(min(1.0, max(0.0, record.probability)), 4),
        signals=signals,
    )
    return EventEnvelope(
        correlation_id=main.incident_id,  # one incident = one ordered Kafka partition stream
        trace_id=f"TRACE-{main.incident_id}",
        source=SOURCE,
        target=TARGET,
        topic=ANOMALY_TOPIC,
        message_type="ANOMALY_DETECTED",
        priority=PRIORITY_BY_SEVERITY[main.severity],
        severity=main.severity,
        timestamp=now or datetime.now(UTC),
        environment=environment,
        service=ServiceIdentity(
            service_id=record.target, namespace=main.namespace, version=version
        ),
        evidence_refs=tuple(sorted(i.incident_id for i in evidence)),
        confidence=round(min(1.0, max(0.0, record.probability)), 4),
        # Same alert sent twice (retry, restart) = same key: consumers drop the duplicate.
        # A new symptom on the same incident is a new situation, hence a new key.
        idempotency_key=f"{SOURCE}:anomaly:{main.incident_id}:{'+'.join(categories)}",
        payload=payload.model_dump(mode="json"),
    )


def build_detection_event(
    anomaly: AnomalyDetected,
    service: ServiceIdentity,
    environment: Environment,
    correlation_id: str,
    severity: Severity,
    now: datetime | None = None,
) -> EventEnvelope:
    """Contract event for an anomaly found on request (a `detect_anomaly` task)."""
    return EventEnvelope(
        correlation_id=correlation_id,
        trace_id=f"TRACE-{correlation_id}",
        source=SOURCE,
        target=TARGET,
        topic=ANOMALY_TOPIC,
        message_type="ANOMALY_DETECTED",
        priority=PRIORITY_BY_SEVERITY[severity],
        severity=severity,
        timestamp=now or datetime.now(UTC),
        environment=environment,
        service=service,
        confidence=anomaly.anomaly_score,
        idempotency_key=f"{SOURCE}:anomaly:{correlation_id}:{anomaly.anomaly_id}",
        payload=anomaly.model_dump(mode="json"),
    )


def build_recovery_event(
    result: RecoveryVerified,
    service: ServiceIdentity,
    environment: Environment,
    correlation_id: str,
    causation_id: str | None = None,
    severity: Severity = Severity.MEDIUM,
    now: datetime | None = None,
) -> EventEnvelope:
    """Contract event for a recovery check of one action (deployment.action.completed)."""
    return EventEnvelope(
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=f"TRACE-{correlation_id}",
        source=SOURCE,
        target=TARGET,
        topic=RECOVERY_TOPIC,
        message_type="RECOVERY_VERIFIED",
        priority=PRIORITY_BY_SEVERITY[severity],
        severity=severity,
        timestamp=now or datetime.now(UTC),
        environment=environment,
        service=service,
        idempotency_key=f"{SOURCE}:recovery:{result.action_id}",
        payload=result.model_dump(mode="json"),
    )


class JsonlEventPublisher:
    """EventPublisher writing one validated event per line, until Kafka is available."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def publish(self, event: EventEnvelope) -> None:
        line = event.model_dump_json() + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append, line)

    def _append(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(line)


def read_events(path: str | Path) -> list[EventEnvelope]:
    """Events of a JSON Lines file, validated again against the contract."""
    file = Path(path)
    if not file.exists():
        return []
    return [
        EventEnvelope.model_validate(json.loads(line))
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class AnomalyEmitter:
    """Publishes the alerts the decision layer chose to notify. Never raises."""

    def __init__(
        self,
        publisher: EventPublisher,
        policy: EventsPolicy | None = None,
        versions: dict[str, str] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._publisher = publisher
        self.policy = policy or EventsPolicy()
        self._versions = versions or {}
        self._clock = clock

    async def emit(
        self,
        record: DecisionRecord | None,
        incidents: list[Incident],
        hypotheses: list[Hypothesis],
    ) -> tuple[EventEnvelope | None, str | None]:
        """(event sent, None), (None, error) or (None, None) when there is nothing to send."""
        if record is None or not record.notify or record.target is None or not incidents:
            return None, None
        try:
            event = build_anomaly_event(
                record,
                incidents,
                hypotheses,
                self.policy.environment,
                self._versions.get(record.target, "unknown"),
                self._clock(),
            )
            await self._publisher.publish(event)
        except Exception as error:  # noqa: BLE001 - a transport failure must not stop the agent
            return None, f"{type(error).__name__}: {error}"
        return event, None


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"
