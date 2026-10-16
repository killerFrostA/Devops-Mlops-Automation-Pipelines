"""Layer 0: a service that stops answering must be an alert, not "insufficient data"."""

import asyncio
from collections.abc import Callable

import httpx
import pytest

from src.agents.monitoring.collectors.profile import Mode, ServiceProfile
from src.agents.monitoring.collectors.prometheus import (
    PrometheusClient,
    PrometheusCollector,
    build_alive_query,
)
from src.agents.monitoring.config import MonitoringConfig, WatchedService
from src.agents.monitoring.decision import Action, DecisionMaker, DecisionPolicy
from src.agents.monitoring.detectors.availability import (
    Availability,
    AvailabilityDetector,
    AvailabilityFinding,
)
from src.agents.monitoring.grouping import (
    Category,
    InMemoryIncidentStore,
    SymptomGrouper,
    from_availability,
)
from src.agents.monitoring.loop import MonitoringLoop
from src.agents.monitoring.reasoning import Reasoner, ReasoningConfig
from src.agents.monitoring.recovery import RecoveryPolicy, Verifier
from src.contracts.enums import Environment, Severity
from src.contracts.tasks import TaskContext
from src.platform.messaging.reliability import RetryPolicy

CONTAINER = ServiceProfile(mode=Mode.CONTAINER, container="fraud-api-1")


def test_queries_use_prometheus_up_or_cadvisor_last_seen() -> None:
    assert (
        build_alive_query("fraud-api", ServiceProfile(), 60)
        == 'max(max_over_time(up{service_id="fraud-api"}[60s]))'
    )
    assert build_alive_query("fraud-api", CONTAINER, 60) == (
        'time() - max(max_over_time(container_last_seen{name="fraud-api-1"}[15m]))'
    )


def prometheus(value: str | None) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        result = [] if value is None else [{"metric": {}, "value": [0, value]}]
        return httpx.Response(
            200, json={"status": "success", "data": {"resultType": "vector", "result": result}}
        )

    return handler


async def no_sleep(_: float) -> None:
    return None


def alive(value: str | None, profile: ServiceProfile | None = None) -> bool | None:
    async def go() -> bool | None:
        transport = httpx.MockTransport(prometheus(value))
        http = httpx.AsyncClient(base_url="http://prom", transport=transport)
        async with PrometheusClient("http://prom", http=http) as client:
            profiles = {"fraud-api": profile} if profile else None
            return await PrometheusCollector(client, profiles=profiles).collect_alive(
                "fraud-api", 60
            )

    return asyncio.run(go())


@pytest.mark.parametrize(
    ("value", "profile", "expected"),
    [
        ("1", None, True),  # at least one scrape answered in the window
        ("0", None, False),  # every scrape failed for 60 s
        (None, None, None),  # never scraped: not configured, not "down"
        ("20", CONTAINER, True),  # container seen 20 s ago
        ("200", CONTAINER, False),  # not seen for 200 s > 60 + cAdvisor margin
        (None, CONTAINER, None),  # not seen for 15 min: unknown
    ],
)
def test_collector_reads_liveness(
    value: str | None, profile: ServiceProfile | None, expected: bool | None
) -> None:
    assert alive(value, profile) is expected


def test_prometheus_down_is_not_a_service_outage() -> None:
    """ "We cannot see" must never be reported as "the service is dead"."""

    def broken(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async def go() -> None:
        http = httpx.AsyncClient(base_url="http://prom", transport=httpx.MockTransport(broken))
        async with PrometheusClient(
            "http://prom",
            http=http,
            retry_policy=RetryPolicy(max_attempts=1),
            sleep=no_sleep,
        ) as client:
            detector = AvailabilityDetector(PrometheusCollector(client))
            await detector.evaluate("fraud-api")

    with pytest.raises(Exception, match="(?i)unavailable|503"):
        asyncio.run(go())


class Alive:
    def __init__(self, answers: list[bool | None]) -> None:
        self.answers = answers

    async def collect_alive(self, service_id: str, window_seconds: int) -> bool | None:
        return self.answers.pop(0)


@pytest.mark.parametrize(
    ("answer", "status"),
    [(True, Availability.UP), (False, Availability.DOWN), (None, Availability.UNKNOWN)],
)
def test_detector_status(answer: bool | None, status: Availability) -> None:
    [finding] = asyncio.run(AvailabilityDetector(Alive([answer])).evaluate("fraud-api"))
    assert (finding.status, finding.window_seconds) == (status, 60)


def test_only_down_becomes_a_high_readiness_symptom() -> None:
    down = AvailabilityFinding(Availability.DOWN, 60)
    [symptom] = from_availability("fraud-api", "local", [down])
    assert (symptom.category, symptom.source, symptom.severity) == (
        Category.READINESS,
        "availability",
        Severity.HIGH,
    )
    assert symptom.facts == {"status": "DOWN", "no_answer_for_seconds": 60}
    for status in (Availability.UP, Availability.UNKNOWN):
        assert from_availability("fraud-api", "local", [AvailabilityFinding(status, 60)]) == []


class Nothing:
    async def evaluate(self, service_id: str) -> list[object]:
        return []  # a dead service: every other detector has no data


def test_a_stopped_service_is_alerted_at_once() -> None:
    loop = MonitoringLoop(
        config=MonitoringConfig(services=(WatchedService(service_id="fraud-api"),)),
        slo=Nothing(),  # type: ignore[arg-type]
        statistical=Nothing(),  # type: ignore[arg-type]
        logs=None,
        grouper=SymptomGrouper(InMemoryIncidentStore()),
        reporter=type("Quiet", (), {"report": lambda self, cycle: None})(),
        reasoner=Reasoner(ReasoningConfig(own_symptom_weights={Category.READINESS: 0.6})),
        decider=DecisionMaker(DecisionPolicy()),
        availability=AvailabilityDetector(Alive([False])),
    )
    cycle = asyncio.run(loop.run_cycle())
    [decision] = cycle.decisions
    assert (decision.incident.category, decision.incident.severity) == (
        Category.READINESS,
        Severity.HIGH,
    )
    assert cycle.decision is not None and cycle.decision.action is Action.ALERT
    assert "regle metier : incident readiness de gravite HIGH" in cycle.decision.reason


class Healthy:
    async def read(self, service_id: str) -> dict[str, float | None]:
        return {"error_rate": 0.0, "latency_p95_ms": 90.0, "memory_slope_mb_min": 0.0}


class Silent:
    async def read(self, service_id: str) -> dict[str, float | None]:
        return {"error_rate": None, "latency_p95_ms": None, "memory_slope_mb_min": None}


POLICY = RecoveryPolicy(settle_seconds=0, window_seconds=120, sample_seconds=30)
CONTEXT = TaskContext(
    service_id="fraud-api",
    namespace="local",
    environment=Environment.LOCAL,
    signals={"categories": ["readiness"]},
)


def test_restart_that_did_not_bring_the_service_back_is_not_recovered() -> None:
    verifier = Verifier(Silent(), POLICY, sleep=no_sleep, alive=Alive([False] * 4))
    result = asyncio.run(verifier.verify("ACT-restart", CONTEXT))
    assert result.checks["service_up"] == "FAIL"
    assert result.status == "NOT_RECOVERED"  # before: INCONCLUSIVE, the outage was invisible


def test_restart_that_worked_is_recovered() -> None:
    verifier = Verifier(Healthy(), POLICY, sleep=no_sleep, alive=Alive([True] * 4))
    result = asyncio.run(verifier.verify("ACT-restart", CONTEXT))
    assert result.checks["service_up"] == "PASS"
    assert result.status == "RECOVERED"
