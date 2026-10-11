import asyncio

import pytest

from src.agents.monitoring.decision import (
    Action,
    DecisionMaker,
    DecisionPolicy,
    DecisionRecord,
    ambiguity,
    score_actions,
)
from src.agents.monitoring.grouping import Category, Incident
from src.agents.monitoring.reasoning import Evidence, Hypothesis
from src.contracts.enums import Severity

POLICY = DecisionPolicy()


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


class Reader:
    def __init__(self, signals: dict[str, dict[str, float | None]], fail: bool = False) -> None:
        self.signals = signals
        self.fail = fail
        self.calls: list[str] = []

    async def read(self, service_id: str) -> dict[str, float | None]:
        self.calls.append(service_id)
        if self.fail:
            raise ConnectionError("prometheus down")
        return self.signals.get(service_id, {})


def hyp(service: str, p: float) -> Hypothesis:
    return Hypothesis(service, p, ())


def incident(severity: Severity = Severity.MEDIUM) -> Incident:
    return Incident(
        "INC-1",
        "s:prod:error_rate",
        "payment-service",
        "prod",
        Category.ERROR_RATE,
        0,
        0,
        1,
        severity,
        frozenset({"slo"}),
        {},
    )


def decide(
    maker: DecisionMaker, hypotheses: list[Hypothesis], incidents: list[Incident] | None = None
) -> DecisionRecord:
    record = asyncio.run(maker.decide(hypotheses, incidents or [incident()]))
    assert record is not None
    return record


def test_confident_agent_alerts_directly() -> None:
    reader = Reader({})
    record = decide(DecisionMaker(POLICY, reader), [hyp("payments-db", 0.92), hyp("api", 0.58)])
    assert record.action is Action.ALERT
    assert record.target == "payments-db"
    assert record.notify
    assert reader.calls == []  # no need to collect more


def test_hesitating_agent_collects_then_alerts_when_confirmed() -> None:
    reader = Reader({"payment-service": {"error_rate": 0.2, "latency_p95_ms": 90.0}})
    record = decide(DecisionMaker(POLICY, reader), [hyp("payment-service", 0.5)])
    assert reader.calls == ["payment-service"]
    assert record.investigations[0].outcome == "confirmed"
    assert record.action is Action.ALERT
    assert record.probability == pytest.approx(1 - 0.5 * 0.4)  # noisy-OR with confirm 0.6


def test_hesitating_agent_waits_when_collection_does_not_confirm() -> None:
    healthy = {"error_rate": 0.001, "latency_p95_ms": 80.0, "memory_slope_mb_min": 0.0}
    reader = Reader({"payment-service": healthy})
    record = decide(DecisionMaker(POLICY, reader), [hyp("payment-service", 0.5)])
    assert record.investigations[0].outcome == "not_confirmed"
    assert record.action is Action.WAIT
    assert record.probability == pytest.approx(0.25)
    assert not record.notify


def test_unavailable_extra_data_is_inconclusive_not_a_crash() -> None:
    record = decide(DecisionMaker(POLICY, Reader({}, fail=True)), [hyp("payment-service", 0.5)])
    assert record.investigations[0].outcome == "inconclusive"
    assert record.probability == 0.5


def test_weak_evidence_means_wait() -> None:
    record = decide(DecisionMaker(POLICY, Reader({})), [hyp("payment-service", 0.15)])
    assert record.action is Action.WAIT


def test_two_close_suspects_trigger_collection_on_both() -> None:
    reader = Reader(
        {"payments-db": {"memory_slope_mb_min": 30.0}, "payment-api": {"error_rate": 0.0}}
    )
    record = decide(
        DecisionMaker(POLICY, reader), [hyp("payment-api", 0.62), hyp("payments-db", 0.6)]
    )
    assert reader.calls == ["payment-api", "payments-db"]
    assert record.target == "payments-db"  # confirmed suspect overtakes the other
    assert record.action is Action.ALERT


def test_business_rule_high_severity_always_alerts() -> None:
    reader = Reader({})
    record = decide(
        DecisionMaker(POLICY, reader), [hyp("payment-service", 0.1)], [incident(Severity.HIGH)]
    )
    assert record.action is Action.ALERT
    assert "regle metier" in record.reason
    assert reader.calls == []


def test_same_alert_is_not_repeated_within_the_window() -> None:
    clock = Clock()
    maker = DecisionMaker(POLICY, Reader({}), clock=clock)
    first = decide(maker, [hyp("payments-db", 0.95)])
    clock.now += 60
    again = decide(maker, [hyp("payments-db", 0.95)])
    clock.now += POLICY.realert_after_seconds
    later = decide(maker, [hyp("payments-db", 0.95)])
    assert (first.notify, again.notify, later.notify) == (True, False, True)


def test_no_hypothesis_no_decision() -> None:
    assert asyncio.run(DecisionMaker(POLICY).decide([], [])) is None


def test_scores_and_ambiguity() -> None:
    assert ambiguity([hyp("a", 0.5)]) == pytest.approx(1)
    assert ambiguity([hyp("a", 0.95)]) == pytest.approx(0.1)
    assert ambiguity([hyp("a", 0.6), hyp("b", 0.6)]) == pytest.approx(1)
    scores = score_actions(0.92, 0.16, POLICY)
    assert max(scores, key=lambda a: scores[a]) is Action.ALERT
    scores = score_actions(0.5, 1.0, POLICY)
    assert max(scores, key=lambda a: scores[a]) is Action.COLLECT_MORE


def hyp_with(service: str, p: float, *categories: str) -> Hypothesis:
    evidence = tuple(Evidence("own_symptom", service, c, 0.5, "sur le service") for c in categories)
    return Hypothesis(service, p, evidence)


def test_new_kind_of_symptom_is_alerted_even_after_a_recent_alert() -> None:
    """Seen live: a false CPU alert on payment-service hid the real memory fault after it."""
    clock = Clock()
    maker = DecisionMaker(POLICY, Reader({}), clock=clock)
    first = decide(maker, [hyp_with("payment-service", 0.9, "cpu_pressure")])
    clock.now += 90
    same = decide(maker, [hyp_with("payment-service", 0.9, "cpu_pressure")])
    clock.now += 90
    memory = decide(maker, [hyp_with("payment-service", 0.9, "cpu_pressure", "memory_pressure")])
    assert (first.notify, same.notify, memory.notify) == (True, False, True)
    assert "nouveau symptome : memory_pressure" in memory.reason


def test_clearly_rising_probability_is_alerted_again() -> None:
    clock = Clock()
    maker = DecisionMaker(POLICY, Reader({}), clock=clock)
    decide(maker, [hyp_with("payment-service", 0.75, "error_rate")])
    clock.now += 60
    slightly = decide(maker, [hyp_with("payment-service", 0.8, "error_rate")])
    clock.now += 60
    clearly = decide(maker, [hyp_with("payment-service", 0.93, "error_rate")])
    assert (slightly.notify, clearly.notify) == (False, True)
    assert "probabilite en hausse" in clearly.reason


def test_business_rule_names_the_newest_incident_when_severity_is_equal() -> None:
    """Seen live (20:10:36): the tail of an earlier latency fault (HIGH, seen 3 times) was
    named instead of the new error-rate incident (HIGH, seen 2 times)."""
    old_latency = Incident(
        "INC-old",
        "payment-service:local:latency",
        "payment-service",
        "local",
        Category.LATENCY,
        100.0,
        200.0,
        3,
        Severity.HIGH,
        frozenset({"slo", "statistical"}),
        {},
    )
    new_errors = Incident(
        "INC-new",
        "payment-service:local:error_rate",
        "payment-service",
        "local",
        Category.ERROR_RATE,
        180.0,
        200.0,
        2,
        Severity.HIGH,
        frozenset({"slo"}),
        {},
    )
    for incidents in ([old_latency, new_errors], [new_errors, old_latency]):  # order-proof
        record = decide(DecisionMaker(POLICY, Reader({})), [hyp("payment-service", 0.7)], incidents)
        assert "incident error_rate de gravite HIGH" in record.reason
