from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agents.monitoring.grouping import (
    Category,
    Incident,
    InMemoryIncidentStore,
    Symptom,
    SymptomGrouper,
)
from src.agents.monitoring.reasoning import (
    LogHint,
    Reasoner,
    ReasoningConfig,
    ServiceNode,
    load_reasoning_config,
    noisy_or,
)
from src.contracts.enums import Severity

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ReasoningConfig(
    window_seconds=300,
    services={
        "frontend": ServiceNode(depends_on=("payment-api",)),
        "payment-api": ServiceNode(depends_on=("payments-db",)),
        "payments-db": ServiceNode(),
    },
    own_symptom_weights={
        Category.MEMORY_PRESSURE: 0.6,
        Category.LATENCY: 0.4,
        Category.LOG_ERRORS: 0.3,
        Category.ERROR_RATE: 0.5,
    },
    dependent_symptom_weight=0.35,
    dependency_log_hints=(LogHint(pattern="pool exhausted|connection pool", weight=0.5),),
)


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def incidents_for(clock: Clock, *symptoms: Symptom) -> list[Incident]:
    grouper = SymptomGrouper(InMemoryIncidentStore(clock), clock=clock)
    return [d.incident for d in grouper.ingest(list(symptoms))]


def symptom(service: str, category: Category, source: str = "statistical", **facts: str) -> Symptom:
    return Symptom(service, "prod", category, source, Severity.MEDIUM, dict(facts))


def test_noisy_or_combines_independent_evidence() -> None:
    assert noisy_or([]) == 0
    assert noisy_or([0.6]) == pytest.approx(0.6)
    assert noisy_or([0.6, 0.5]) == pytest.approx(0.8)  # 1 - 0.4 * 0.5


def test_demo_scenario_points_to_the_database() -> None:
    """API latency + new "connection pool exhausted" log on the API + abnormal database RAM:
    the database, which the API depends on, must come out as the most probable origin."""
    clock = Clock()
    reasoner = Reasoner(GRAPH, clock=clock)
    reasoner.observe(
        incidents_for(
            clock,
            symptom("payment-api", Category.LATENCY),
            symptom(
                "payment-api",
                Category.LOG_ERRORS,
                "logs",
                template="ERROR HikariPool-1 connection pool exhausted <*>",
            ),
            symptom("payments-db", Category.MEMORY_PRESSURE),
        )
    )
    first, second = reasoner.rank()[:2]
    assert first.service_id == "payments-db"
    assert second.service_id == "payment-api"
    assert first.probability > second.probability
    # 1 - (1-0.6 own RAM)(1-0.35 API latency)(1-0.35 API log)(1-0.5 pool hint)
    assert first.probability == pytest.approx(1 - 0.4 * 0.65 * 0.65 * 0.5)
    assert {e.kind for e in first.evidence} == {
        "own_symptom",
        "dependent_symptom",
        "dependency_log_hint",
    }


def test_symptom_on_a_leaf_service_only_accuses_itself() -> None:
    clock = Clock()
    reasoner = Reasoner(GRAPH, clock=clock)
    reasoner.observe(incidents_for(clock, symptom("payments-db", Category.MEMORY_PRESSURE)))
    ranked = reasoner.rank()
    assert [h.service_id for h in ranked] == ["payments-db"]
    assert ranked[0].probability == pytest.approx(0.6)


def test_caller_symptom_also_raises_suspicion_on_its_dependency() -> None:
    clock = Clock()
    reasoner = Reasoner(GRAPH, clock=clock)
    reasoner.observe(incidents_for(clock, symptom("payment-api", Category.ERROR_RATE)))
    probabilities = {h.service_id: h.probability for h in reasoner.rank()}
    assert probabilities == {"payment-api": pytest.approx(0.5), "payments-db": 0.35}


def test_unknown_service_is_reasoned_on_its_own_symptoms() -> None:
    clock = Clock()
    reasoner = Reasoner(GRAPH, clock=clock)
    reasoner.observe(incidents_for(clock, symptom("payment-service", Category.LATENCY)))
    assert [h.service_id for h in reasoner.rank()] == ["payment-service"]


def test_old_incidents_leave_the_window() -> None:
    clock = Clock()
    reasoner = Reasoner(GRAPH, clock=clock)
    reasoner.observe(incidents_for(clock, symptom("payments-db", Category.MEMORY_PRESSURE)))
    clock.now += 301
    assert reasoner.rank() == []


def test_the_same_incident_seen_again_counts_once() -> None:
    clock = Clock()
    grouper = SymptomGrouper(InMemoryIncidentStore(clock), clock=clock)
    reasoner = Reasoner(GRAPH, clock=clock)
    for _ in range(5):
        reasoner.observe(
            [d.incident for d in grouper.ingest([symptom("payments-db", Category.MEMORY_PRESSURE)])]
        )
        clock.now += 60
    assert reasoner.rank()[0].probability == pytest.approx(0.6)


def test_graph_errors_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown"):
        ReasoningConfig(services={"a": ServiceNode(depends_on=("ghost",))})
    with pytest.raises(ValidationError, match="itself"):
        ReasoningConfig(services={"a": ServiceNode(depends_on=("a",))})
    with pytest.raises(ValidationError):
        LogHint(pattern="(unclosed", weight=0.5)
    with pytest.raises(ValidationError):
        ReasoningConfig(own_symptom_weights={Category.LATENCY: 1.5})


def test_project_graph_file_is_valid() -> None:
    config = load_reasoning_config(ROOT / "configs/dependencies.yaml")
    assert config.dependencies("payment-api") == ("payments-db",)
    assert config.own_symptom_weights[Category.MEMORY_OOM] == 0.9
