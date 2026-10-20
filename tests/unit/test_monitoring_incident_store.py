"""Open incidents and recent alerts survive an agent restart."""

import asyncio
from pathlib import Path

from src.agents.monitoring.decision import DecisionMaker, DecisionPolicy
from src.agents.monitoring.grouping import Action, Category, Incident, Symptom, SymptomGrouper
from src.agents.monitoring.incident_store import SqliteIncidentStore
from src.agents.monitoring.memory import AlertMemory
from src.agents.monitoring.reasoning import Evidence, Hypothesis
from src.contracts.enums import Severity


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


LEAK = Symptom(
    "payment-service",
    "local",
    Category.MEMORY_PRESSURE,
    "statistical",
    Severity.MEDIUM,
    {"slope_per_minute": 30.0, "nan": None},
)


def test_incident_round_trips_through_sqlite() -> None:
    clock = Clock()
    store = SqliteIncidentStore(":memory:", clock)
    [decision] = SymptomGrouper(store, clock=clock).ingest([LEAK])
    assert store.get(LEAK.signature) == decision.incident  # every field, sources and facts too
    clock.now += 901
    assert store.get(LEAK.signature) is None  # expired after the cooldown


def test_restarted_agent_continues_the_same_incident(tmp_path: Path) -> None:
    path = tmp_path / "incidents.sqlite"
    clock = Clock()
    first_store = SqliteIncidentStore(path, clock)
    [opened] = SymptomGrouper(first_store, clock=clock).ingest([LEAK])
    first_store.close()  # the agent stops...

    clock.now += 60  # ...and restarts a minute later, the leak still going
    second_store = SqliteIncidentStore(path, clock)
    [again] = SymptomGrouper(second_store, clock=clock).ingest([LEAK])
    second_store.close()
    assert (opened.action, again.action) == (Action.OPENED, Action.UPDATED)
    assert again.incident.incident_id == opened.incident.incident_id
    assert again.incident.occurrences == 2
    assert not again.notify  # no duplicate notification after the restart


def leak_hypothesis() -> Hypothesis:
    evidence = (
        Evidence("own_symptom", "payment-service", "memory_pressure", 0.6, "sur le service"),
    )
    return Hypothesis("payment-service", 0.9, evidence)


def incident(clock: Clock) -> Incident:
    return Incident(
        "INC-1",
        LEAK.signature,
        "payment-service",
        "local",
        Category.MEMORY_PRESSURE,
        clock.now - 600,
        clock.now,
        5,
        Severity.MEDIUM,
        frozenset({"statistical"}),
        {},
    )


def test_restarted_agent_does_not_send_a_recent_alert_again(tmp_path: Path) -> None:
    clock = Clock()
    memory = AlertMemory(tmp_path / "memory.sqlite", clock=clock)
    before = DecisionMaker(DecisionPolicy(), clock=clock, memory=memory)
    first = asyncio.run(before.decide([leak_hypothesis()], [incident(clock)]))
    assert first is not None and first.notify

    clock.now += 120  # restart: a new DecisionMaker, same memory file
    after = DecisionMaker(DecisionPolicy(), clock=clock, memory=memory)
    again = asyncio.run(after.decide([leak_hypothesis()], [incident(clock)]))
    assert again is not None and not again.notify
    assert "deja alerte" in again.reason
    memory.close()


def test_old_alerts_are_not_restored(tmp_path: Path) -> None:
    clock = Clock()
    memory = AlertMemory(tmp_path / "memory.sqlite", clock=clock)
    asyncio.run(
        DecisionMaker(DecisionPolicy(), clock=clock, memory=memory).decide(
            [leak_hypothesis()], [incident(clock)]
        )
    )
    clock.now += DecisionPolicy().realert_after_seconds + 60  # restart long after
    later = asyncio.run(
        DecisionMaker(DecisionPolicy(), clock=clock, memory=memory).decide(
            [leak_hypothesis()], [incident(clock)]
        )
    )
    assert later is not None and later.notify
    memory.close()
