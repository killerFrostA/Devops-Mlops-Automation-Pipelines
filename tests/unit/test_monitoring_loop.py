import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agents.monitoring.config import MonitoringConfig, WatchedService, load_config
from src.agents.monitoring.detectors.logs import FindingKind, LogFinding, LogLabel
from src.agents.monitoring.detectors.slo import SLOEvaluation, SLOStatus
from src.agents.monitoring.detectors.statistical import FindingStatus, StatisticalFinding
from src.agents.monitoring.grouping import Action, Category, InMemoryIncidentStore, SymptomGrouper
from src.agents.monitoring.loop import ConsoleReporter, CycleReport, MonitoringLoop
from src.contracts.enums import Severity

ROOT = Path(__file__).resolve().parents[2]
MEMORY_GROWTH = StatisticalFinding(
    "memory_mb",
    "memory_pressure",
    "growth",
    FindingStatus.ANOMALY,
    slope_per_minute=30.0,
    severity=Severity.MEDIUM,
)
ERROR_BREACH = SLOEvaluation(
    "availability", "fast-burn", SLOStatus.BREACH, 41.5, 48.6, 14.4, Severity.HIGH
)


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


class Fake[T]:
    """Detector returning scripted results per service; can fail or hang for one service."""

    def __init__(self, results: dict[str, list[T]] | None = None) -> None:
        self.results = results or {}
        self.fail_for: set[str] = set()
        self.hang_for: set[str] = set()
        self.calls = 0

    async def evaluate(self, service_id: str) -> list[T]:
        self.calls += 1
        if service_id in self.fail_for:
            raise ConnectionError("source unreachable")
        if service_id in self.hang_for:
            await asyncio.sleep(3600)
        return self.results.get(service_id, [])


class Collect:
    def __init__(self) -> None:
        self.cycles: list[CycleReport] = []

    def report(self, cycle: CycleReport) -> None:
        self.cycles.append(cycle)


def make_loop(
    clock: Clock,
    slo: Fake[SLOEvaluation] | None = None,
    statistical: Fake[StatisticalFinding] | None = None,
    logs: Fake[LogFinding] | None = None,
    services: tuple[str, ...] = ("payment-service",),
    **kwargs: object,
) -> tuple[MonitoringLoop, Collect]:
    config = MonitoringConfig(
        interval_seconds=60,
        service_timeout_seconds=0.2,
        services=tuple(WatchedService(service_id=s) for s in services),
    )
    reporter = Collect()
    loop = MonitoringLoop(
        config=config,
        slo=slo or Fake(),
        statistical=statistical or Fake(),
        logs=logs,
        grouper=SymptomGrouper(InMemoryIncidentStore(clock), clock=clock),
        reporter=reporter,
        clock=clock,
        **kwargs,  # type: ignore[arg-type]
    )
    return loop, reporter


def test_a_cycle_turns_detections_into_an_incident_on_its_own() -> None:
    loop, reporter = make_loop(Clock(), statistical=Fake({"payment-service": [MEMORY_GROWTH]}))
    cycle = asyncio.run(loop.run_cycle())
    assert [d.action for d in cycle.decisions] == [Action.OPENED]
    assert cycle.notifications[0].incident.category is Category.MEMORY_PRESSURE
    assert reporter.cycles == [cycle]
    assert cycle.duration_seconds >= 0


def test_ten_minutes_of_the_same_fault_is_one_incident_notified_once() -> None:
    clock = Clock()
    loop, reporter = make_loop(clock, statistical=Fake({"payment-service": [MEMORY_GROWTH]}))

    async def ten_cycles() -> None:
        for _ in range(10):
            await loop.run_cycle()
            clock.now += 60

    asyncio.run(ten_cycles())
    ids = {d.incident.incident_id for c in reporter.cycles for d in c.decisions}
    assert len(ids) == 1
    assert sum(len(c.notifications) for c in reporter.cycles) == 1


def test_every_detector_contributes_and_sources_confirm_one_incident() -> None:
    log = LogFinding(
        FindingKind.NEW_PATTERN,
        "ERROR Unhandled exception <*>",
        LogLabel.HTTP_SERVER_ERROR,
        "regex",
        60,
        0,
        "x",
        False,
    )
    loop, _ = make_loop(
        Clock(),
        slo=Fake({"payment-service": [ERROR_BREACH]}),
        statistical=Fake({"payment-service": [MEMORY_GROWTH]}),
        logs=Fake({"payment-service": [log]}),
    )
    cycle = asyncio.run(loop.run_cycle())
    by_category = {d.incident.category: d.incident for d in cycle.decisions}
    assert set(by_category) == {Category.ERROR_RATE, Category.MEMORY_PRESSURE}
    assert by_category[Category.ERROR_RATE].sources == {"slo", "logs"}


def test_a_failing_detector_does_not_stop_the_others() -> None:
    slo: Fake[SLOEvaluation] = Fake()
    slo.fail_for.add("payment-service")
    loop, _ = make_loop(Clock(), slo=slo, statistical=Fake({"payment-service": [MEMORY_GROWTH]}))
    cycle = asyncio.run(loop.run_cycle())
    assert "slo" in cycle.services[0].errors
    assert len(cycle.decisions) == 1  # the statistical detector still worked


def test_a_hanging_service_times_out_without_blocking_the_other_service() -> None:
    statistical: Fake[StatisticalFinding] = Fake(
        {"payment-service": [MEMORY_GROWTH], "order-service": [MEMORY_GROWTH]}
    )
    statistical.hang_for.add("order-service")
    loop, _ = make_loop(
        Clock(), statistical=statistical, services=("payment-service", "order-service")
    )
    cycle = asyncio.run(loop.run_cycle())
    reports = {s.service_id: s for s in cycle.services}
    assert "timeout" in reports["order-service"].errors
    assert reports["payment-service"].errors == {}
    assert [d.incident.service_id for d in cycle.decisions] == ["payment-service"]


def test_unavailable_source_is_reported() -> None:
    async def down() -> bool:
        return False

    async def broken() -> bool:
        raise ConnectionError

    async def up() -> bool:
        return True

    loop, _ = make_loop(Clock(), health_checks={"prometheus": down, "loki": broken, "x": up})
    cycle = asyncio.run(loop.run_cycle())
    assert cycle.sources_down == ("prometheus", "loki")


def test_logs_disabled_in_config_are_not_called() -> None:
    logs: Fake[LogFinding] = Fake()
    loop, _ = make_loop(Clock(), logs=logs)
    loop.config = loop.config.model_copy(update={"logs_enabled": False})
    asyncio.run(loop.run_cycle())
    assert logs.calls == 0


def test_run_forever_runs_cycles_then_stops_cleanly() -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    loop, reporter = make_loop(Clock(), sleep=fake_sleep)
    asyncio.run(loop.run_forever(asyncio.Event(), max_cycles=3))
    assert len(reporter.cycles) == 3
    assert all(0 < s <= 60 for s in slept)  # waits the rest of the interval


def test_stop_event_interrupts_the_wait() -> None:
    async def scenario() -> int:
        stop = asyncio.Event()
        loop, reporter = make_loop(Clock())  # real sleep of ~60 s, must be interrupted
        task = asyncio.create_task(loop.run_forever(stop))
        await asyncio.sleep(0.05)
        stop.set()
        await asyncio.wait_for(task, timeout=2)
        return len(reporter.cycles)

    assert asyncio.run(scenario()) == 1


def test_console_report_is_readable() -> None:
    lines: list[str] = []
    loop, _ = make_loop(Clock(), statistical=Fake({"payment-service": [MEMORY_GROWTH]}))
    loop.reporter = ConsoleReporter(lines.append)
    asyncio.run(loop.run_cycle())
    assert lines[0].startswith("[") and "cycle" in lines[0]
    assert any("INCIDENT OUVERT" in line and "memory_pressure" in line for line in lines)


def test_project_config_file_is_valid() -> None:
    config = load_config(ROOT / "configs/monitoring.yaml")
    assert config.services[0].service_id == "payment-service"
    rules = {rule.name: rule.burn_rate for rule in config.slo.rules}
    assert (rules["very-fast"], rules["fast-burn"], rules["slow-burn"]) == (36, 14.4, 6.0)


def test_config_rejects_incoherent_values(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="shorter"):
        MonitoringConfig(interval_seconds=30, service_timeout_seconds=30)
    with pytest.raises(ValidationError, match="once"):
        MonitoringConfig(services=(WatchedService(service_id="a"), WatchedService(service_id="a")))
    bad = tmp_path / "bad.yaml"
    bad.write_text("interval_seconds: 60\nunknown_key: 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_config(empty).interval_seconds == 60


def test_a_hanging_probe_is_bounded_by_the_timeout() -> None:
    async def hangs() -> bool:
        await asyncio.sleep(3600)
        return True

    loop, _ = make_loop(Clock(), health_checks={"prometheus": hangs})
    cycle = asyncio.run(asyncio.wait_for(loop.run_cycle(), timeout=2))
    assert cycle.sources_down == ("prometheus",)
    assert cycle.duration_seconds < 1


def test_verbose_report_shows_every_layer_even_when_normal() -> None:
    normal_slo = SLOEvaluation("availability", "fast-burn", SLOStatus.OK, 0.2, 0.1, 14.4, None)
    normal_stat = StatisticalFinding(
        "latency_p95_ms",
        "latency",
        "level_shift",
        FindingStatus.NORMAL,
        current=90.0,
        baseline_median=89.0,
        robust_z=0.4,
    )
    lines: list[str] = []
    loop, _ = make_loop(
        Clock(),
        slo=Fake({"payment-service": [normal_slo]}),
        statistical=Fake({"payment-service": [normal_stat]}),
        logs=Fake(),
    )
    loop.reporter = ConsoleReporter(lines.append, verbose=True)
    asyncio.run(loop.run_cycle())
    text = "\n".join(lines)
    assert "SLO" in text and "-> OK" in text
    assert "Statistique" in text and "-> NORMAL" in text
    assert "Logs" in text and "aucun modele nouveau" in text
    assert "INCIDENT" not in text


def test_non_verbose_report_stays_quiet_when_normal() -> None:
    lines: list[str] = []
    loop, _ = make_loop(Clock())
    loop.reporter = ConsoleReporter(lines.append)
    asyncio.run(loop.run_cycle())
    assert len(lines) == 1
