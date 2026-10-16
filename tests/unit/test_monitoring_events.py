import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]

from src.agents.monitoring.decision import Action, DecisionMaker, DecisionPolicy, DecisionRecord
from src.agents.monitoring.events import (
    AnomalyEmitter,
    EventsPolicy,
    JsonlEventPublisher,
    build_anomaly_event,
    build_recovery_event,
    read_events,
)
from src.agents.monitoring.grouping import Category, Incident
from src.agents.monitoring.reasoning import Evidence, Hypothesis
from src.contracts.enums import Environment, Priority, Severity
from src.contracts.events import EventEnvelope, ServiceIdentity
from src.contracts.payloads import AnomalyDetected, RecoveryVerified

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 10, 11, 18, 24, 3, tzinfo=UTC)


def schema() -> Draft202012Validator:
    path = ROOT / "contracts/jsonschema/v1/EventEnvelope.schema.json"
    return Draft202012Validator(json.loads(path.read_text()), format_checker=FormatChecker())


def incident(
    service: str = "payment-service",
    category: Category = Category.MEMORY_PRESSURE,
    severity: Severity = Severity.MEDIUM,
    incident_id: str = "INC-2026-5173687a6c",
) -> Incident:
    return Incident(
        incident_id,
        f"{service}:local:{category}",
        service,
        "local",
        category,
        1_791_735_813.0,
        1_791_735_843.0,
        3,
        severity,
        frozenset({"statistical", "forecast"}),
        {"statistical.slope_per_minute": 22.6, "forecast.current": 76.73, "x.nan": float("nan")},
    )


def alert(
    target: str = "payment-service", categories: frozenset[str] | None = None
) -> DecisionRecord:
    return DecisionRecord(
        Action.ALERT,
        target,
        0.84,
        {},
        "apres collecte : payment-service confirmed",
        notify=True,
        categories=categories or frozenset({"memory_pressure"}),
        alert_id=1,
    )


HYPOTHESES = [Hypothesis("payment-service", 0.6, ())]


def event(**kwargs: object) -> EventEnvelope:
    defaults: dict[str, object] = {
        "record": alert(),
        "incidents": [incident()],
        "hypotheses": HYPOTHESES,
        "environment": Environment.LOCAL,
        "now": NOW,
    }
    defaults.update(kwargs)
    return build_anomaly_event(**defaults)  # type: ignore[arg-type]


def test_alert_becomes_a_valid_v1_contract_event() -> None:
    sent = event()
    document = sent.model_dump(mode="json")
    schema().validate(document)  # the team's JSON Schema
    assert EventEnvelope.model_validate(document) == sent  # and the Python contract
    assert (sent.topic, sent.message_type) == ("monitoring.anomaly.detected", "ANOMALY_DETECTED")
    assert (sent.source, sent.target) == ("monitoring-agent", "orchestrator")
    assert sent.correlation_id == sent.partition_key == "INC-2026-5173687a6c"
    assert (sent.priority, sent.severity, sent.confidence) == (Priority.P2, Severity.MEDIUM, 0.84)
    payload = AnomalyDetected.model_validate(sent.payload)
    assert payload.detector == "forecast+statistical"
    assert payload.anomaly_score == 0.84
    assert payload.signals["categories"] == ["memory_pressure"]


def test_payload_states_facts_never_a_root_cause() -> None:
    text = event().model_dump_json().lower()
    assert "root_cause" not in text and "cause" not in text
    signals = event().payload["signals"]
    assert isinstance(signals, dict)
    assert signals["suspects"] == [{"service_id": "payment-service", "probability": 0.6}]


def test_non_finite_facts_become_null_so_the_json_stays_valid() -> None:
    document = json.loads(event().model_dump_json())  # what the other agents receive
    facts = document["payload"]["signals"]["incidents"][0]["facts"]
    assert facts["x.nan"] is None
    assert facts["statistical.slope_per_minute"] == 22.6


@pytest.mark.parametrize(
    ("severity", "priority"),
    [
        (Severity.CRITICAL, Priority.P0),
        (Severity.HIGH, Priority.P1),
        (Severity.MEDIUM, Priority.P2),
        (Severity.LOW, Priority.P3),
    ],
)
def test_priority_follows_the_worst_severity(severity: Severity, priority: Priority) -> None:
    assert event(incidents=[incident(severity=severity)]).priority is priority


def test_same_alert_same_key_new_symptom_new_key() -> None:
    first, retried = event(), event(now=NOW.replace(minute=30))
    assert first.idempotency_key == retried.idempotency_key
    assert first.message_id != retried.message_id
    # A new symptom is a new incident observed on the service.
    errors = incident(category=Category.ERROR_RATE, incident_id="INC-errors")
    worse = event(
        record=alert(categories=frozenset({"memory_pressure", "error_rate"})),
        incidents=[incident(), errors],
    )
    assert worse.idempotency_key != first.idempotency_key


def test_suspect_without_own_incident_carries_the_incidents_seen_upstream() -> None:
    api_errors = incident("payment-api", Category.ERROR_RATE, incident_id="INC-api")
    sent = event(record=alert("payments-db", frozenset({"error_rate"})), incidents=[api_errors])
    assert sent.service.service_id == "payments-db"
    assert sent.evidence_refs == ("INC-api",)
    signals = sent.payload["signals"]
    assert isinstance(signals, dict) and signals["observed_on"] == ["payment-api"]


def test_recovery_result_becomes_a_valid_v1_contract_event() -> None:
    result = RecoveryVerified(
        verification_id="VER-1",
        action_id="ACT-1",
        window="5m",
        status="RECOVERED",
        checks={"error_rate": "PASS", "latency_p95_ms": "PASS", "memory_slope_mb_min": "UNKNOWN"},
    )
    sent = build_recovery_event(
        result,
        ServiceIdentity(service_id="payment-service", namespace="local", version="v2"),
        Environment.LOCAL,
        correlation_id="INC-1",
        causation_id="MSG-action",
        now=NOW,
    )
    schema().validate(sent.model_dump(mode="json"))
    assert (sent.topic, sent.causation_id) == ("monitoring.recovery.verified", "MSG-action")
    assert sent.idempotency_key == "monitoring-agent:recovery:ACT-1"


def test_jsonl_publisher_writes_events_that_read_back_valid(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    publisher = JsonlEventPublisher(path)
    asyncio.run(publisher.publish(event()))
    asyncio.run(publisher.publish(event(incidents=[incident(severity=Severity.HIGH)])))
    events = read_events(path)
    assert [e.priority for e in events] == [Priority.P2, Priority.P1]
    for line in path.read_text(encoding="utf-8").splitlines():
        schema().validate(json.loads(line))


class Failing:
    async def publish(self, event: EventEnvelope) -> None:
        raise ConnectionError("kafka down")


class Collecting:
    def __init__(self) -> None:
        self.events: list[EventEnvelope] = []

    async def publish(self, event: EventEnvelope) -> None:
        self.events.append(event)


def test_emitter_sends_only_new_alerts_and_never_raises() -> None:
    sink = Collecting()
    emitter = AnomalyEmitter(sink, EventsPolicy(), {"payment-service": "v2"}, clock=lambda: NOW)
    sent, error = asyncio.run(emitter.emit(alert(), [incident()], HYPOTHESES))
    assert sent is not None and error is None and sink.events == [sent]
    assert sent.service.version == "v2"
    repeated = DecisionRecord(Action.ALERT, "payment-service", 0.84, {}, "deja alerte")
    assert asyncio.run(emitter.emit(repeated, [incident()], HYPOTHESES)) == (None, None)
    assert asyncio.run(emitter.emit(None, [], [])) == (None, None)

    broken = AnomalyEmitter(Failing())
    sent, error = asyncio.run(broken.emit(alert(), [incident()], HYPOTHESES))
    assert sent is None and error == "ConnectionError: kafka down"


def test_a_real_decision_reaches_the_other_agents() -> None:
    hypothesis = Hypothesis(
        "payment-service",
        0.9,
        (Evidence("own_symptom", "payment-service", "memory_pressure", 0.6, "sur le service"),),
    )
    record = asyncio.run(DecisionMaker(DecisionPolicy()).decide([hypothesis], [incident()]))
    sink = Collecting()
    sent, _ = asyncio.run(AnomalyEmitter(sink).emit(record, [incident()], [hypothesis]))
    assert sent is not None
    schema().validate(sent.model_dump(mode="json"))


def test_full_cycle_detection_to_event(tmp_path: Path) -> None:
    from src.agents.monitoring.config import MonitoringConfig, WatchedService
    from src.agents.monitoring.detectors.statistical import FindingStatus, StatisticalFinding
    from src.agents.monitoring.grouping import InMemoryIncidentStore, SymptomGrouper
    from src.agents.monitoring.loop import ConsoleReporter, MonitoringLoop
    from src.agents.monitoring.reasoning import Reasoner, ReasoningConfig

    leak = StatisticalFinding(
        "memory_mb", "memory_pressure", "growth", FindingStatus.ANOMALY, slope_per_minute=30.0
    )

    class Fixed:
        def __init__(self, results: list[StatisticalFinding]) -> None:
            self.results = results

        async def evaluate(self, service_id: str) -> list[StatisticalFinding]:
            return self.results

    lines: list[str] = []
    path = tmp_path / "events.jsonl"
    loop = MonitoringLoop(
        config=MonitoringConfig(services=(WatchedService(service_id="payment-service"),)),
        slo=Fixed([]),  # type: ignore[arg-type]
        statistical=Fixed([leak]),
        logs=None,
        grouper=SymptomGrouper(InMemoryIncidentStore()),
        reporter=ConsoleReporter(lines.append),
        reasoner=Reasoner(ReasoningConfig(own_symptom_weights={Category.MEMORY_PRESSURE: 0.9})),
        decider=DecisionMaker(DecisionPolicy()),
        emitter=AnomalyEmitter(JsonlEventPublisher(path)),
    )
    cycle = asyncio.run(loop.run_cycle())
    assert cycle.event is not None and cycle.event_error is None
    assert read_events(path) == [cycle.event]
    assert any("=> EVENEMENT monitoring.anomaly.detected -> orchestrator" in line for line in lines)
    again = asyncio.run(loop.run_cycle())  # same situation: alert not repeated, nothing sent
    assert again.event is None and len(read_events(path)) == 1


def test_event_is_about_the_newest_incident_when_severity_is_equal() -> None:
    old = incident(category=Category.LATENCY, severity=Severity.HIGH, incident_id="INC-old")
    new = Incident(
        "INC-new",
        "payment-service:local:error_rate",
        "payment-service",
        "local",
        Category.ERROR_RATE,
        old.opened_at + 300,
        old.last_seen_at,
        1,  # seen fewer times than the old one, but it is the new situation
        Severity.HIGH,
        frozenset({"slo"}),
        {},
    )
    sent = event(
        record=alert(categories=frozenset({"latency", "error_rate"})), incidents=[old, new]
    )
    assert sent.correlation_id == "INC-new"
    assert AnomalyDetected.model_validate(sent.payload).anomaly_id.startswith("ANM-INC-new-")


def test_higher_severity_is_a_new_situation_for_the_consumers() -> None:
    medium, high = event(), event(incidents=[incident(severity=Severity.HIGH)])
    assert medium.idempotency_key != high.idempotency_key  # not dropped as a duplicate
    assert (medium.priority, high.priority) == (Priority.P2, Priority.P1)


def test_event_lists_only_what_is_observed_now() -> None:
    """Seen live on fraud-api (14:42:32): the service was back UP since 14:40, but the alert
    still listed `readiness`, kept by the 5-minute reasoning window."""
    errors = incident(category=Category.ERROR_RATE)
    sent = event(
        record=alert(categories=frozenset({"error_rate", "readiness"})), incidents=[errors]
    )
    signals = sent.payload["signals"]
    assert isinstance(signals, dict)
    assert signals["categories"] == ["error_rate"]
    assert sent.idempotency_key.endswith(":error_rate:MEDIUM")
