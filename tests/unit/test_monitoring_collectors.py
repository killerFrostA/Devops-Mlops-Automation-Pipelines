import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pytest

from src.agents.monitoring.collectors.models import MetricSnapshot
from src.agents.monitoring.collectors.prometheus import (
    PrometheusClient,
    PrometheusCollector,
    PrometheusQueryError,
    build_queries,
)
from src.platform.messaging.reliability import RetryPolicy
from src.platform.resilience import CircuitBreaker
from src.platform.resilience import CircuitState as State

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)

# Sample answers keyed by a fragment unique to each query.
HEALTHY = {
    "sum(process_resident_memory_bytes": "250.1",
    "deriv(": "59.8",
    "process_cpu_seconds_total": "0.12",
    'sum(rate(http_requests_total{service_id="payment-service",path!~"/health.*|/metrics|'
    '/chaos.*"}[1m]))': "4.9",
    "or vector(0)": "0.02",
    "histogram_quantile": "90",
}


def vector(value: str) -> dict[str, object]:
    return {
        "status": "success",
        "data": {"resultType": "vector", "result": [{"metric": {}, "value": [0, value]}]},
    }


def prometheus_answering(
    answers: dict[str, str | None],
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params["query"]
        for fragment, value in answers.items():
            if fragment in query:
                if value is None:
                    empty = {"status": "success", "data": {"resultType": "vector", "result": []}}
                    return httpx.Response(200, json=empty)
                return httpx.Response(200, json=vector(value))
        raise AssertionError(f"unexpected query: {query}")

    return handler


async def no_sleep(_: float) -> None:
    return None


def collect(
    handler: Callable[[httpx.Request], httpx.Response], breaker: CircuitBreaker | None = None
) -> MetricSnapshot:
    async def run() -> MetricSnapshot:
        http = httpx.AsyncClient(base_url="http://prom", transport=httpx.MockTransport(handler))
        async with PrometheusClient(
            "http://prom",
            http=http,
            retry_policy=RetryPolicy(max_attempts=2),
            breaker=breaker,
            sleep=no_sleep,
        ) as client:
            return await PrometheusCollector(client, clock=lambda: NOW).collect("payment-service")

    return asyncio.run(run())


def test_collects_every_signal_into_a_typed_snapshot() -> None:
    snapshot = collect(prometheus_answering(HEALTHY))
    assert snapshot.memory_mb == 250.1
    assert snapshot.memory_slope_mb_min == 59.8
    assert snapshot.request_rate_rps == 4.9
    assert snapshot.error_rate == 0.02
    assert snapshot.latency_p95_ms == 90
    assert snapshot.collected_at == NOW
    assert snapshot.complete


def test_empty_or_nan_series_are_unknown_not_zero() -> None:
    answers: dict[str, str | None] = {**HEALTHY, "histogram_quantile": "NaN", "deriv(": None}
    snapshot = collect(prometheus_answering(answers))
    assert snapshot.latency_p95_ms is None
    assert snapshot.memory_slope_mb_min is None
    assert set(snapshot.missing) == {"latency_p95_ms", "memory_slope_mb_min"}
    assert snapshot.source_available
    assert not snapshot.complete


def test_transient_5xx_is_retried() -> None:
    attempts: list[str] = []
    healthy = prometheus_answering(HEALTHY)

    def handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params["query"]
        if "histogram_quantile" in query and query not in attempts:
            attempts.append(query)
            return httpx.Response(503)
        return healthy(request)

    assert collect(handler).latency_p95_ms == 90


def test_prometheus_down_degrades_without_crashing() -> None:
    def down(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    breaker = CircuitBreaker(failure_threshold=1)
    snapshot = collect(down, breaker=breaker)
    assert not snapshot.source_available
    assert len(snapshot.missing) == len(build_queries("payment-service"))
    assert breaker.state is State.OPEN


def test_rejected_query_is_raised_not_hidden() -> None:
    def rejecting(_: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"status": "error", "error": "parse error"})

    with pytest.raises(PrometheusQueryError):
        collect(rejecting)


@pytest.mark.parametrize("service_id", ['x"} or vector(1) or {a="', "Payment", "", "a_b"])
def test_invalid_service_id_is_refused(service_id: str) -> None:
    async def run() -> None:
        async with PrometheusClient("http://prom") as client:
            await PrometheusCollector(client).collect(service_id)

    with pytest.raises(ValueError, match="Invalid service_id"):
        asyncio.run(run())
