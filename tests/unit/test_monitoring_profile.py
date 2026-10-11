"""Agent 1 applies to any project: metric names and the way of measuring come from config."""

import asyncio
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from src.agents.monitoring.collectors.loki import LokiClient
from src.agents.monitoring.collectors.profile import MetricNames, Mode, ServiceProfile
from src.agents.monitoring.collectors.prometheus import (
    PrometheusClient,
    PrometheusCollector,
    build_queries,
    build_series_queries,
    build_sli_queries,
)
from src.agents.monitoring.config import MonitoringConfig, WatchedService, load_config
from src.agents.monitoring.detectors.slo import SLOEvaluation
from src.agents.monitoring.detectors.statistical import FindingStatus, StatisticalFinding
from src.agents.monitoring.grouping import InMemoryIncidentStore, SymptomGrouper
from src.agents.monitoring.loop import MonitoringLoop

CONTAINER = ServiceProfile(mode=Mode.CONTAINER, container="nexusfx-visual-api")


def test_default_profile_keeps_the_standard_queries() -> None:
    queries = build_queries("payment-service")
    assert set(queries) == {
        "memory_mb",
        "memory_slope_mb_min",
        "cpu_cores",
        "request_rate_rps",
        "error_rate",
        "latency_p95_ms",
    }
    assert (
        'http_requests_total{service_id="payment-service",path!~"/health.*'
        in (queries["request_rate_rps"])
    )
    assert 'status=~"5.."' in queries["error_rate"]


def test_another_project_with_other_metric_names() -> None:
    names = MetricNames(
        requests="api_calls_total",
        latency_histogram="api_call_seconds",
        status_label="code",
        path_label="handler",
        excluded_paths="",
    )
    queries = build_queries("order-api", ServiceProfile(metrics=names))
    assert 'api_calls_total{service_id="order-api",code=~"5.."}' in queries["error_rate"]
    assert "api_call_seconds_bucket" in queries["latency_p95_ms"]
    assert "handler" not in queries["request_rate_rps"]  # empty filter: all traffic counts
    sli = build_sli_queries("order-api", "5m", "0.5", ServiceProfile(metrics=names))
    assert 'api_call_seconds_bucket{service_id="order-api",le="0.5"}' in sli["slow_ratio"]


def test_container_mode_measures_from_outside_with_cadvisor() -> None:
    queries = build_queries("visual-api", CONTAINER)
    assert set(queries) == {"memory_mb", "memory_slope_mb_min", "cpu_cores"}  # no traffic
    assert 'container_memory_working_set_bytes{name="nexusfx-visual-api"}' in queries["memory_mb"]
    # Seen live: cAdvisor samples every 30 s, so a 1m rate had no value and CPU stayed unknown.
    assert (
        queries["cpu_cores"]
        == 'sum(rate(container_cpu_usage_seconds_total{name="nexusfx-visual-api"}[2m]))'
    )
    assert set(build_series_queries("visual-api", CONTAINER)) == {"memory_mb", "cpu_cores"}
    assert build_sli_queries("visual-api", "5m", "0.5", CONTAINER) == {}


def test_container_name_defaults_to_the_service_id() -> None:
    queries = build_queries("visual-api", ServiceProfile(mode=Mode.CONTAINER))
    assert 'name="visual-api"' in queries["memory_mb"]


@pytest.mark.parametrize(
    "bad",
    [
        {"metrics": {"requests": 'x{a="b"}'}},
        {"metrics": {"status_label": "code-x"}},
        {"metrics": {"excluded_paths": '/health"} or vector(1) #'}},
        {"metrics": {"excluded_paths": "("}},
        {"container": 'api"}'},
    ],
)
def test_names_cannot_inject_promql(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ServiceProfile.model_validate(bad)


def answering(values: dict[str, str]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params["query"]
        value = next(v for fragment, v in values.items() if fragment in query)
        result = [{"metric": {}, "value": [0, value]}]
        return httpx.Response(
            200, json={"status": "success", "data": {"resultType": "vector", "result": result}}
        )

    return httpx.AsyncClient(base_url="http://prom", transport=httpx.MockTransport(handler))


def test_collector_uses_each_service_profile() -> None:
    async def go() -> None:
        http = answering({"container_memory": "100", "container_cpu": "0.25"})
        async with PrometheusClient("http://prom", http=http) as client:
            collector = PrometheusCollector(client, profiles={"visual-api": CONTAINER})
            snapshot = await collector.collect("visual-api")
            assert (snapshot.memory_mb, snapshot.cpu_cores) == (100.0, 0.25)
            assert snapshot.error_rate is None
            assert snapshot.missing == ()  # not measured is not "missing"
            assert snapshot.source_available
            windows = await collector.collect_sli("visual-api", ["5m"], "0.5")
            assert windows["5m"].request_rate_rps is None
            assert windows["5m"].source_available

    asyncio.run(go())


def test_logs_of_a_container_are_read_by_its_name() -> None:
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        queries.append(request.url.params["query"])
        return httpx.Response(
            200, json={"status": "success", "data": {"resultType": "streams", "result": []}}
        )

    async def go() -> None:
        http = httpx.AsyncClient(base_url="http://loki", transport=httpx.MockTransport(handler))
        containers = {"visual-api": "nexusfx-visual-api"}
        async with LokiClient("http://loki", http=http, containers=containers) as client:
            await client.query_lines("visual-api", 0, 1)
            await client.query_lines("payment-service", 0, 1)

    asyncio.run(go())
    assert queries == ['{container="nexusfx-visual-api"}', '{service_id="payment-service"}']


class Recorder[T]:
    def __init__(self, results: list[T] | None = None) -> None:
        self.seen: list[str] = []
        self.results = results or []

    async def evaluate(self, service_id: str) -> list[T]:
        self.seen.append(service_id)
        return self.results


class Quiet:
    def report(self, cycle: object) -> None:
        return None


def test_loop_skips_slo_for_a_container_but_watches_its_resources() -> None:
    memory = StatisticalFinding(
        "memory_mb", "memory_pressure", "growth", FindingStatus.ANOMALY, slope_per_minute=20.0
    )
    config = MonitoringConfig(
        services=(
            WatchedService(service_id="payment-service"),
            WatchedService(service_id="visual-api", mode=Mode.CONTAINER),
        )
    )
    slo: Recorder[SLOEvaluation] = Recorder()
    statistical = Recorder([memory])
    loop = MonitoringLoop(
        config=config,
        slo=slo,
        statistical=statistical,
        logs=None,
        grouper=SymptomGrouper(InMemoryIncidentStore()),
        reporter=Quiet(),
    )
    cycle = asyncio.run(loop.run_cycle())
    assert slo.seen == ["payment-service"]
    assert sorted(statistical.seen) == ["payment-service", "visual-api"]
    assert {d.incident.service_id for d in cycle.decisions} == {"payment-service", "visual-api"}


def test_yaml_declares_services_of_any_project(tmp_path: Path) -> None:
    path = tmp_path / "monitoring.yaml"
    path.write_text(
        "services:\n"
        "  - service_id: payment-service\n"
        "  - service_id: visual-api\n"
        "    mode: container\n"
        "    container: nexusfx-visual-api\n"
        "  - service_id: order-api\n"
        "    metrics: {requests: api_calls_total, status_label: code}\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert [s.mode for s in config.services] == [Mode.APP, Mode.CONTAINER, Mode.APP]
    assert config.services[2].metrics.requests == "api_calls_total"
    assert config.containers() == {"visual-api": "nexusfx-visual-api"}
