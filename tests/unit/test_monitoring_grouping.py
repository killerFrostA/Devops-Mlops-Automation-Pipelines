from datetime import UTC, datetime

import pytest

from src.agents.monitoring.detectors.logs import FindingKind, LogFinding, LogLabel
from src.agents.monitoring.detectors.slo import SLOEvaluation, SLOStatus
from src.agents.monitoring.detectors.statistical import FindingStatus, StatisticalFinding
from src.agents.monitoring.grouping import (
    Action,
    Category,
    InMemoryIncidentStore,
    Symptom,
    SymptomGrouper,
    from_logs,
    from_slo,
    from_statistical,
)
from src.contracts.enums import Severity

T0 = datetime(2026, 10, 7, 9, 50, tzinfo=UTC).timestamp()


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> float:
        return self.now


def grouper(clock: Clock, store: InMemoryIncidentStore | None = None) -> SymptomGrouper:
    counter = iter(range(1, 1000))
    return SymptomGrouper(
        store or InMemoryIncidentStore(clock),
        cooldown_seconds=900,
        clock=clock,
        id_factory=lambda _: f"INC-{next(counter)}",
    )


def symptom(
    category: Category = Category.MEMORY_PRESSURE,
    source: str = "statistical",
    severity: Severity = Severity.MEDIUM,
    service: str = "payment-service",
) -> Symptom:
    return Symptom(service, "prod", category, source, severity, {"value": 1.0})


def test_one_long_problem_is_one_incident_and_notifies_once() -> None:
    clock = Clock()
    g = grouper(clock)
    decisions = []
    for _ in range(40):  # every 15 s for 10 minutes
        decisions += g.ingest([symptom()])
        clock.now += 15
    assert {d.incident.incident_id for d in decisions} == {"INC-1"}
    assert [d.action for d in decisions].count(Action.OPENED) == 1
    assert sum(d.notify for d in decisions) == 1
    assert decisions[-1].incident.occurrences == 40


def test_problem_crossing_the_hour_stays_one_incident() -> None:
    clock = Clock()  # 09:50
    g = grouper(clock)
    first = g.ingest([symptom()])[0]
    for minute in (54, 58):  # still observed, every 4 min
        clock.now = datetime(2026, 10, 7, 9, minute, tzinfo=UTC).timestamp()
        g.ingest([symptom()])
    clock.now = datetime(2026, 10, 7, 10, 1, tzinfo=UTC).timestamp()  # past 10:00
    second = g.ingest([symptom()])[0]
    assert second.action is Action.UPDATED
    assert second.incident.incident_id == first.incident.incident_id


def test_different_symptoms_at_the_same_time_are_separate_incidents() -> None:
    g = grouper(Clock())
    decisions = g.ingest([symptom(Category.MEMORY_PRESSURE), symptom(Category.NETWORK_ERRORS)])
    assert len({d.incident.incident_id for d in decisions}) == 2
    assert all(d.action is Action.OPENED for d in decisions)


def test_same_symptom_on_different_services_are_separate_incidents() -> None:
    g = grouper(Clock())
    decisions = g.ingest([symptom(service="payment-service"), symptom(service="order-service")])
    assert len({d.incident.incident_id for d in decisions}) == 2


def test_flicker_with_short_gaps_is_the_same_incident() -> None:
    clock = Clock()
    g = grouper(clock)
    first = g.ingest([symptom()])[0]
    clock.now += 240  # 4 min of silence: the symptom flickers, below the 5 min reopen delay
    again = g.ingest([symptom()])[0]
    assert again.incident.incident_id == first.incident.incident_id
    assert not again.notify


def test_return_after_cooldown_is_a_new_incident() -> None:
    clock = Clock()
    g = grouper(clock)
    first = g.ingest([symptom()])[0]
    clock.now += 901
    again = g.ingest([symptom()])[0]
    assert again.action is Action.OPENED
    assert again.incident.incident_id != first.incident.incident_id


def test_escalation_and_new_source_notify() -> None:
    clock = Clock()
    g = grouper(clock)
    g.ingest([symptom(Category.ERROR_RATE, "slo", Severity.MEDIUM)])
    clock.now += 15
    escalated = g.ingest([symptom(Category.ERROR_RATE, "slo", Severity.HIGH)])[0]
    assert escalated.notify and escalated.incident.severity is Severity.HIGH
    clock.now += 15
    confirmed = g.ingest([symptom(Category.ERROR_RATE, "logs", Severity.MEDIUM)])[0]
    assert confirmed.notify
    assert confirmed.incident.sources == {"slo", "logs"}
    assert confirmed.incident.severity is Severity.HIGH  # severity never goes down
    assert set(confirmed.incident.facts) == {"slo.value", "logs.value"}


def test_same_signature_twice_in_one_batch_gives_one_decision() -> None:
    g = grouper(Clock())
    decisions = g.ingest([symptom(source="statistical"), symptom(source="logs")])
    assert len(decisions) == 1
    assert decisions[0].action is Action.OPENED and decisions[0].notify
    assert decisions[0].incident.occurrences == 2


def test_restart_with_a_shared_store_does_not_duplicate() -> None:
    clock = Clock()
    store = InMemoryIncidentStore(clock)
    first = grouper(clock, store).ingest([symptom()])[0]
    clock.now += 30
    after_restart = grouper(clock, store).ingest([symptom()])[0]  # new grouper, same store
    assert after_restart.incident.incident_id == first.incident.incident_id
    assert after_restart.action is Action.UPDATED


def test_cooldown_must_be_positive() -> None:
    with pytest.raises(ValueError):
        SymptomGrouper(InMemoryIncidentStore(), cooldown_seconds=0)


def test_adapters_keep_only_abnormal_results() -> None:
    slo = [
        SLOEvaluation(
            "availability", "fast-burn", SLOStatus.BREACH, 41.5, 48.6, 14.4, Severity.HIGH
        ),
        SLOEvaluation("latency-500ms", "fast-burn", SLOStatus.OK, 0.0, 0.0, 14.4, None),
        SLOEvaluation(
            "availability", "slow-burn", SLOStatus.INSUFFICIENT_DATA, None, None, 6, None
        ),
    ]
    slo_symptoms = from_slo(
        "payment-service", "prod", {"availability": "availability", "latency-500ms": "latency"}, slo
    )
    assert [(s.category, s.severity) for s in slo_symptoms] == [
        (Category.ERROR_RATE, Severity.HIGH)
    ]

    stats = [
        StatisticalFinding(
            "memory_mb",
            "memory_pressure",
            "growth",
            FindingStatus.ANOMALY,
            slope_per_minute=30.0,
            severity=Severity.MEDIUM,
        ),
        StatisticalFinding("cpu_cores", "cpu_pressure", "level_shift", FindingStatus.NORMAL),
    ]
    assert [s.category for s in from_statistical("payment-service", "prod", stats)] == [
        Category.MEMORY_PRESSURE
    ]

    def log(label: LogLabel) -> LogFinding:
        return LogFinding(
            FindingKind.NEW_PATTERN, "ERROR x <*>", label, "regex", 340, 0, "x", False
        )

    categories = [
        s.category
        for s in from_logs(
            "payment-service",
            "prod",
            [log(LogLabel.CONNECTION_REFUSED), log(LogLabel.OUT_OF_MEMORY), log(LogLabel.UNKNOWN)],
        )
    ]
    # An "out of memory" log line is memory pressure: memory_oom needs an observed OOM kill.
    assert categories == [Category.NETWORK_ERRORS, Category.MEMORY_PRESSURE, Category.LOG_ERRORS]


def test_slo_and_log_evidence_of_errors_join_one_incident() -> None:
    """The SLO breach and the matching 5xx log template describe the same symptom category."""
    g = grouper(Clock())
    breach = SLOEvaluation(
        "availability", "fast-burn", SLOStatus.BREACH, 41.5, 48.6, 14.4, Severity.HIGH
    )
    template = LogFinding(
        FindingKind.NEW_PATTERN,
        "ERROR Unhandled exception <*>",
        LogLabel.HTTP_SERVER_ERROR,
        "regex",
        60,
        0,
        "x",
        False,
    )
    symptoms = from_slo("payment-service", "prod", {"availability": "availability"}, [breach])
    symptoms += from_logs("payment-service", "prod", [template])
    decisions = g.ingest(symptoms)
    assert len(decisions) == 1
    assert decisions[0].incident.sources == {"slo", "logs"}


def test_a_symptom_back_after_a_quiet_period_is_a_new_incident() -> None:
    """Seen live (reasoning test, 19:59): a new latency fault 12 min after the previous one was
    merged into the old incident, so the orchestrator was never told."""
    from src.agents.monitoring.grouping import Category, InMemoryIncidentStore, Symptom

    now = [1_000.0]
    grouper = SymptomGrouper(
        InMemoryIncidentStore(lambda: now[0]),
        cooldown_seconds=900,
        clock=lambda: now[0],
        reopen_after_seconds=300,
    )
    slow = Symptom("payment-service", "local", Category.LATENCY, "slo", Severity.HIGH)
    [first] = grouper.ingest([slow])
    now[0] += 60
    [still] = grouper.ingest([slow])  # same fault, still going
    now[0] += 420  # 7 min without the symptom: the fault had stopped
    [again] = grouper.ingest([slow])
    assert (first.action, still.action, again.action) == (
        Action.OPENED,
        Action.UPDATED,
        Action.OPENED,
    )
    assert still.incident.incident_id == first.incident.incident_id
    assert again.incident.incident_id != first.incident.incident_id
    assert again.notify
