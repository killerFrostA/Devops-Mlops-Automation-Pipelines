"""memory_oom: a container killed by the kernel for lack of memory is a CRITICAL (P0) alert."""

import asyncio
from pathlib import Path

import pytest

from src.agents.monitoring.collectors.profile import Mode, ServiceProfile
from src.agents.monitoring.collectors.prometheus import (
    build_memory_peak_ratio_query,
    build_oom_query,
    build_restart_query,
)
from src.agents.monitoring.config import MonitoringConfig, WatchedService
from src.agents.monitoring.decision import Action, DecisionMaker, DecisionPolicy
from src.agents.monitoring.detectors.oom import OomDetector, OomFinding
from src.agents.monitoring.events import AnomalyEmitter, JsonlEventPublisher, read_events
from src.agents.monitoring.grouping import (
    Category,
    InMemoryIncidentStore,
    SymptomGrouper,
    from_oom,
)
from src.agents.monitoring.loop import MonitoringLoop
from src.agents.monitoring.reasoning import Reasoner, ReasoningConfig
from src.contracts.enums import Priority, Severity


def test_oom_query_finds_the_container_by_label_or_name() -> None:
    assert build_oom_query("payment-service", ServiceProfile(), 300) == (
        'sum(increase(container_oom_events_total{container_label_service_id="payment-service"}'
        "[300s]))"
    )
    container = ServiceProfile(mode=Mode.CONTAINER, container="nexusfx-api")
    assert 'name="nexusfx-api"' in build_oom_query("visual-api", container, 300)


def test_restart_and_memory_peak_queries() -> None:
    assert build_restart_query("payment-service", ServiceProfile(), 300) == (
        'sum(changes(process_start_time_seconds{service_id="payment-service"}[300s]))'
    )
    container = ServiceProfile(mode=Mode.CONTAINER, container="nexusfx-api")
    assert build_restart_query("visual-api", container, 300) is None
    ratio = build_memory_peak_ratio_query("payment-service", ServiceProfile(), 300)
    assert "max_over_time(container_memory_working_set_bytes" in ratio
    assert "container_spec_memory_limit_bytes" in ratio and "> 0" in ratio


class Kills:
    def __init__(
        self, value: float | None, restarts: float | None = 0.0, ratio: float | None = None
    ) -> None:
        self.value = value
        self.restarts = restarts
        self.ratio = ratio

    async def collect_oom_kills(self, service_id: str, window_seconds: int) -> float | None:
        return self.value

    async def collect_restarts(self, service_id: str, window_seconds: int) -> float | None:
        return self.restarts

    async def collect_memory_peak_ratio(self, service_id: str, window_seconds: int) -> float | None:
        return self.ratio


@pytest.mark.parametrize(("kills", "killed"), [(1.0, True), (0.0, False), (None, False)])
def test_only_an_observed_kill_counts(kills: float | None, killed: bool) -> None:
    [finding] = asyncio.run(OomDetector(Kills(kills)).evaluate("payment-service"))
    assert finding.killed is killed
    symptoms = from_oom("payment-service", "local", [finding])
    assert len(symptoms) == int(killed)


def test_oom_symptom_is_critical() -> None:
    [symptom] = from_oom("payment-service", "local", [OomFinding(2.0, 300)])
    assert (symptom.category, symptom.source, symptom.severity) == (
        Category.MEMORY_OOM,
        "events",
        Severity.CRITICAL,
    )
    assert symptom.facts["oom_kills"] == 2.0


# Docker Desktop / WSL2: the kernel counter stays at 0, the restart at the limit is the proof.
@pytest.mark.parametrize(
    ("restarts", "ratio", "category"),
    [
        (1.0, 1.0, Category.MEMORY_OOM),  # restarted with memory at the limit: OOM kill
        (1.0, 0.95, Category.MEMORY_OOM),
        (1.0, 0.4, Category.READINESS),  # restarted, memory fine: unplanned restart
        (1.0, None, Category.READINESS),  # no memory limit: cannot say OOM
        (0.0, 1.0, None),  # memory high but no restart: not a kill (memory_pressure's job)
        (None, None, None),  # container mode, not measured
    ],
)
def test_restart_at_the_memory_limit_is_an_oom_kill(
    restarts: float | None, ratio: float | None, category: Category | None
) -> None:
    [finding] = asyncio.run(OomDetector(Kills(0.0, restarts, ratio)).evaluate("payment-service"))
    symptoms = from_oom("payment-service", "local", [finding])
    assert [s.category for s in symptoms] == ([category] if category else [])
    if category is Category.MEMORY_OOM:
        assert symptoms[0].severity is Severity.CRITICAL
    if category is Category.READINESS:
        assert symptoms[0].severity is Severity.HIGH


def test_memory_peak_is_only_read_after_a_restart_or_kill() -> None:
    class Counting(Kills):
        asked = 0

        async def collect_memory_peak_ratio(
            self, service_id: str, window_seconds: int
        ) -> float | None:
            self.asked += 1
            return 1.0

    quiet = Counting(0.0, 0.0)
    asyncio.run(OomDetector(quiet).evaluate("payment-service"))
    assert quiet.asked == 0


class Nothing:
    async def evaluate(self, service_id: str) -> list[object]:
        return []


class Quiet:
    def report(self, cycle: object) -> None:
        return None


def test_oom_kill_reaches_the_orchestrator_as_p0(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    loop = MonitoringLoop(
        config=MonitoringConfig(services=(WatchedService(service_id="payment-service"),)),
        slo=Nothing(),  # type: ignore[arg-type]
        statistical=Nothing(),  # type: ignore[arg-type]
        logs=None,
        grouper=SymptomGrouper(InMemoryIncidentStore()),
        reporter=Quiet(),
        reasoner=Reasoner(ReasoningConfig(own_symptom_weights={Category.MEMORY_OOM: 0.9})),
        decider=DecisionMaker(DecisionPolicy()),
        emitter=AnomalyEmitter(JsonlEventPublisher(path)),
        oom=OomDetector(Kills(1.0)),
    )
    cycle = asyncio.run(loop.run_cycle())
    assert cycle.decision is not None and cycle.decision.action is Action.ALERT
    assert "memory_oom de gravite CRITICAL" in cycle.decision.reason
    [event] = read_events(path)
    assert (event.priority, event.severity) == (Priority.P0, Severity.CRITICAL)
