import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]

from src.agents.monitoring.config import MonitoringConfig, WatchedService
from src.agents.monitoring.detectors.statistical import FindingStatus, StatisticalFinding
from src.agents.monitoring.grouping import InMemoryIncidentStore, SymptomGrouper
from src.agents.monitoring.loop import MonitoringLoop
from src.agents.monitoring.service import MonitoringAgent, OneShotDetector
from src.contracts.enums import AgentId, Environment, ResultStatus, Severity
from src.contracts.events import EventEnvelope
from src.contracts.payloads import AnomalyDetected, RecoveryVerified
from src.contracts.tasks import AgentTask, TaskContext

ROOT = Path(__file__).resolve().parents[2]


def validate(event: EventEnvelope) -> None:
    path = ROOT / "contracts/jsonschema/v1/EventEnvelope.schema.json"
    Draft202012Validator(json.loads(path.read_text()), format_checker=FormatChecker()).validate(
        event.model_dump(mode="json")
    )


def task(objective: str, signals: dict[str, str] | None = None, **changes: object) -> AgentTask:
    built = AgentTask(
        task_id="TASK-1",
        incident_id="INC-1",
        agent_id=AgentId.MONITORING,
        objective=objective,
        deadline=datetime.now(UTC) + timedelta(minutes=5),
        task_context=TaskContext(
            service_id="payment-service",
            namespace="local",
            environment=Environment.LOCAL,
            signals=dict(signals or {}),
        ),
    )
    return built.model_copy(update=changes)


class Collecting:
    def __init__(self) -> None:
        self.events: list[EventEnvelope] = []

    async def publish(self, event: EventEnvelope) -> None:
        self.events.append(event)


class Found:
    last_severity = Severity.HIGH

    def __init__(self, anomaly: AnomalyDetected | None) -> None:
        self.anomaly = anomaly

    async def detect(self, context: TaskContext) -> AnomalyDetected | None:
        return self.anomaly


class Checked:
    def __init__(self) -> None:
        self.asked: list[str] = []

    async def verify(self, action_id: str, context: TaskContext) -> RecoveryVerified:
        self.asked.append(action_id)
        return RecoveryVerified(
            verification_id="VER-1",
            action_id=action_id,
            window="5m",
            status="RECOVERED",
            checks={"memory_slope_mb_min": "PASS"},
        )


ANOMALY = AnomalyDetected(
    anomaly_id="ANM-1",
    detector="forecast+statistical",
    anomaly_score=0.84,
    signals={"categories": ["memory_pressure"]},
)


def test_not_wired_agent_stays_honest_skeleton() -> None:
    result = asyncio.run(MonitoringAgent().handle(task("detect_anomaly")))
    assert result.status is ResultStatus.NOT_IMPLEMENTED


def test_detect_task_reports_and_publishes_the_anomaly() -> None:
    sink = Collecting()
    agent = MonitoringAgent(Found(ANOMALY), Checked(), sink, {"payment-service": "v2"})
    result = asyncio.run(agent.handle(task("detect_anomaly")))
    assert result.status is ResultStatus.SUCCEEDED
    assert (result.confidence, result.evidence_refs) == (0.84, ("ANM-1",))
    assert "memory_pressure" in result.rationale_summary
    [event] = sink.events
    validate(event)
    assert (event.topic, event.correlation_id) == ("monitoring.anomaly.detected", "INC-1")
    assert (event.severity, event.service.version) == (Severity.HIGH, "v2")


def test_detect_task_on_a_healthy_service() -> None:
    sink = Collecting()
    result = asyncio.run(MonitoringAgent(Found(None), None, sink).handle(task("detect_anomaly")))
    assert result.status is ResultStatus.SUCCEEDED
    assert result.rationale_summary == "Aucune anomalie sur payment-service"
    assert result.confidence is None and sink.events == []


def test_verify_task_checks_the_named_action_and_publishes_the_result() -> None:
    sink, verifier = Collecting(), Checked()
    agent = MonitoringAgent(None, verifier, sink)
    result = asyncio.run(agent.handle(task("verify_recovery", {"action_id": "ACT-7"})))
    assert result.status is ResultStatus.SUCCEEDED
    assert result.rationale_summary.startswith("RECOVERED apres ACT-7 sur 5m")
    assert verifier.asked == ["ACT-7"]
    [event] = sink.events
    validate(event)
    assert event.topic == "monitoring.recovery.verified"


@pytest.mark.parametrize(
    ("objective", "signals", "message"),
    [
        ("verify_recovery", {}, "signals.action_id"),
        ("restart_service", {}, "non pris en charge"),
    ],
)
def test_bad_tasks_fail_with_a_reason(
    objective: str, signals: dict[str, str], message: str
) -> None:
    result = asyncio.run(MonitoringAgent(Found(None), Checked()).handle(task(objective, signals)))
    assert result.status is ResultStatus.FAILED
    assert message in result.rationale_summary


def test_wired_agent_still_rejects_wrong_identity_and_expired_tasks() -> None:
    agent = MonitoringAgent(Found(None), Checked())
    with pytest.raises(ValueError, match="different agent"):
        asyncio.run(agent.handle(task("detect_anomaly", agent_id=AgentId.DIAGNOSIS)))
    expired = task("detect_anomaly", deadline=datetime.now(UTC) - timedelta(seconds=1))
    with pytest.raises(ValueError, match="expired"):
        asyncio.run(agent.handle(expired))


class Fixed:
    def __init__(self, results: list[StatisticalFinding]) -> None:
        self.results = results
        self.asked: list[str] = []

    async def evaluate(self, service_id: str) -> list[StatisticalFinding]:
        self.asked.append(service_id)
        return self.results


def one_shot(
    statistical: list[StatisticalFinding], forecast: list[StatisticalFinding]
) -> tuple[AnomalyDetected | None, list[str]]:
    stat, fore = Fixed(statistical), Fixed(forecast)
    loop = MonitoringLoop(
        config=MonitoringConfig(services=(WatchedService(service_id="payment-service"),)),
        slo=Fixed([]),  # type: ignore[arg-type]
        statistical=stat,
        logs=None,
        grouper=SymptomGrouper(InMemoryIncidentStore()),
        reporter=None,  # type: ignore[arg-type]  # one-shot detection never reports
        forecast=fore,
    )
    detector = OneShotDetector(loop)
    return asyncio.run(
        detector.detect(
            TaskContext(service_id="checkout", namespace="shop", environment=Environment.LOCAL)
        )
    ), stat.asked


def test_one_shot_detection_runs_every_detector_on_any_service() -> None:
    def leak(method: str) -> StatisticalFinding:
        return StatisticalFinding(
            "memory_mb",
            "memory_pressure",
            method,
            FindingStatus.ANOMALY,
            severity=Severity.MEDIUM,
        )

    anomaly, asked = one_shot([leak("growth")], [leak("forecast")])
    assert asked == ["checkout"]  # not in the config: watched with the defaults
    assert isinstance(anomaly, AnomalyDetected)
    assert anomaly.detector == "forecast+statistical"
    assert anomaly.anomaly_score == pytest.approx(1 - 0.4 * 0.4)  # two agreeing sources
    assert anomaly.signals["categories"] == ["memory_pressure"]


def test_one_shot_detection_on_a_healthy_service_finds_nothing() -> None:
    anomaly, _ = one_shot([], [])
    assert anomaly is None
