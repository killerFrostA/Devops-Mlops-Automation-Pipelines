import asyncio
from pathlib import Path

import pytest

from src.agents.monitoring import feedback
from src.agents.monitoring.decision import Action, DecisionMaker, DecisionPolicy
from src.agents.monitoring.grouping import Category, Incident
from src.agents.monitoring.memory import AlertMemory, MemoryPolicy, Verdict
from src.agents.monitoring.reasoning import Evidence, Hypothesis
from src.contracts.enums import Severity

HEALTHY: dict[str, float | None] = {
    "error_rate": 0.001,
    "latency_p95_ms": 80.0,
    "memory_slope_mb_min": 0.0,
}


class Reader:
    async def read(self, service_id: str) -> dict[str, float | None]:
        return HEALTHY


def hyp(service: str, p: float, *categories: str) -> Hypothesis:
    evidence = tuple(Evidence("own_symptom", service, c, 0.5, "sur le service") for c in categories)
    return Hypothesis(service, p, evidence)


def incident(severity: Severity) -> Incident:
    return Incident(
        "INC-1",
        "s:local:memory_pressure",
        "payment-service",
        "local",
        Category.MEMORY_PRESSURE,
        0,
        0,
        1,
        severity,
        frozenset({"statistical"}),
        {},
    )


def judged(memory: AlertMemory, verdict: Verdict, times: int, category: str) -> None:
    for _ in range(times):
        alert_id = memory.record_alert("payment-service", [category], 0.8, "test")
        memory.add_feedback(alert_id, verdict)


def test_no_feedback_changes_nothing() -> None:
    memory = AlertMemory(":memory:")
    memory.record_alert("payment-service", ["memory_pressure"], 0.8, "assez sur de lui")
    assert memory.adjust("payment-service", ["memory_pressure"], 0.8) == (0.8, [])


def test_false_alerts_make_the_agent_careful_true_ones_confident() -> None:
    memory = AlertMemory(":memory:")
    judged(memory, Verdict.FALSE, 3, "cpu_pressure")
    judged(memory, Verdict.TRUE, 3, "memory_pressure")
    careful, _ = memory.adjust("payment-service", ["cpu_pressure"], 0.8)
    confident, _ = memory.adjust("payment-service", ["memory_pressure"], 0.6)
    # Beta(2,2) + 3 false: precision 2/7, odds x0.4 -> 0.8 becomes 0.615
    assert careful == pytest.approx(1.6 / 2.6)
    # Beta(2,2) + 3 true: precision 5/7, odds x2.5 -> 0.6 becomes 0.789
    assert confident == pytest.approx(3.75 / 4.75)


def test_feedback_is_per_service_and_symptom() -> None:
    memory = AlertMemory(":memory:")
    judged(memory, Verdict.FALSE, 3, "cpu_pressure")
    assert memory.adjust("payment-service", ["memory_pressure"], 0.8)[0] == 0.8
    assert memory.adjust("other-service", ["cpu_pressure"], 0.8)[0] == 0.8


def test_learned_influence_is_bounded() -> None:
    memory = AlertMemory(":memory:", MemoryPolicy(max_odds_factor=4))
    judged(memory, Verdict.FALSE, 50, "cpu_pressure")
    reliability = memory.reliability("payment-service", "cpu_pressure")
    assert reliability.odds_factor == pytest.approx(0.25)


def test_new_verdict_replaces_the_previous_one_and_unknown_alert_is_refused() -> None:
    memory = AlertMemory(":memory:")
    alert_id = memory.record_alert("payment-service", ["error_rate"], 0.9, "x")
    memory.add_feedback(alert_id, Verdict.FALSE)
    memory.add_feedback(alert_id, Verdict.TRUE, "en fait c'etait vrai")
    stored = memory.alerts()[0]
    assert (stored.verdict, stored.note) == (Verdict.TRUE, "en fait c'etait vrai")
    assert memory.reliability("payment-service", "error_rate").false_alerts == 0
    with pytest.raises(KeyError):
        memory.add_feedback(999, Verdict.TRUE)


def test_memory_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite"
    first = AlertMemory(path)
    judged(first, Verdict.FALSE, 2, "cpu_pressure")
    first.close()
    again = AlertMemory(path)
    assert again.reliability("payment-service", "cpu_pressure").false_alerts == 2
    assert len(again.alerts()) == 2
    again.close()


def test_alerts_sent_are_recorded_with_their_number() -> None:
    memory = AlertMemory(":memory:")
    maker = DecisionMaker(DecisionPolicy(), Reader(), memory=memory)
    record = asyncio.run(maker.decide([hyp("payment-service", 0.9, "memory_pressure")], []))
    assert record is not None and record.notify
    assert record.alert_id == 1
    stored = memory.alerts()[0]
    assert (stored.service_id, stored.categories) == ("payment-service", ("memory_pressure",))


def test_after_false_alerts_the_same_evidence_no_longer_alerts() -> None:
    memory = AlertMemory(":memory:")
    first = asyncio.run(
        DecisionMaker(DecisionPolicy(), Reader(), memory=memory).decide(
            [hyp("payment-service", 0.8, "cpu_pressure")], []
        )
    )
    assert first is not None and first.action is Action.ALERT
    # 3 false alerts: 0.8 -> 0.62, still alerted after a check (alerting pays above p=0.30).
    # 6 false alerts: odds divided by 4 -> 0.5 -> check not confirmed -> 0.25 -> wait.
    judged(memory, Verdict.FALSE, 6, "cpu_pressure")
    later = asyncio.run(
        DecisionMaker(DecisionPolicy(), Reader(), memory=memory).decide(
            [hyp("payment-service", 0.8, "cpu_pressure")], []
        )
    )
    assert later is not None
    assert later.action is Action.WAIT  # less sure -> collected -> not confirmed -> wait
    assert "memoire" in later.reason and "6 fausse(s)" in later.reason
    assert later.alert_id is None


def test_business_rule_still_alerts_high_severity_after_false_alerts() -> None:
    # The judged alerts are past ones (dated 0): not recent alerts to restore after a restart.
    memory = AlertMemory(":memory:", clock=lambda: 0.0)
    judged(memory, Verdict.FALSE, 10, "memory_pressure")
    maker = DecisionMaker(DecisionPolicy(), Reader(), memory=memory)
    record = asyncio.run(
        maker.decide([hyp("payment-service", 0.3, "memory_pressure")], [incident(Severity.HIGH)])
    )
    assert record is not None and record.action is Action.ALERT
    assert record.alert_id is not None


def test_feedback_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "monitoring.yaml"
    db = tmp_path / "memory.sqlite"
    config.write_text(f"memory:\n  path: {db.as_posix()}\n", encoding="utf-8")
    memory = AlertMemory(db)
    alert_id = memory.record_alert("payment-service", ["cpu_pressure"], 0.8, "x")
    memory.close()

    feedback.main(["fausse", str(alert_id), "--note", "pic normal", "--config", str(config)])
    feedback.main(["list", "--config", str(config)])
    feedback.main(["stats", "--config", str(config)])
    out = capsys.readouterr().out
    assert "marquee comme fausse" in out
    assert "[fausse]  note : pic normal" in out
    assert "0 vraie(s) 1 fausse(s)" in out
