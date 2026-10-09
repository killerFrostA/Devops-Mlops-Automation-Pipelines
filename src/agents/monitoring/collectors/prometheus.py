"""Prometheus collector: turns PromQL answers into a typed MetricSnapshot.

Read-only. Each query is bounded by a timeout, retried on transient failures and guarded by a
circuit breaker, so an unavailable Prometheus degrades into "unknown" values instead of a crash.
"""

import asyncio
import math
import re
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from typing import Any

import httpx

from src.agents.monitoring.collectors.http import (
    GuardedClient,
    SourceQueryError,
    SourceUnavailableError,
)
from src.agents.monitoring.collectors.models import MetricSnapshot, Series, SLIWindow
from src.platform.messaging.reliability import RetryPolicy
from src.platform.resilience import CircuitBreaker, CircuitOpenError


class PrometheusUnavailableError(SourceUnavailableError):
    """Prometheus unreachable, timed out or 5xx: retried and counted by the circuit breaker."""


class PrometheusQueryError(SourceQueryError):
    """Prometheus rejected the query (4xx or error status): a caller bug, never retried."""


class PrometheusClient(GuardedClient):
    def __init__(
        self,
        base_url: str,
        http: httpx.AsyncClient | None = None,
        timeout_seconds: float = 5.0,
        retry_policy: RetryPolicy | None = None,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__(
            base_url,
            PrometheusUnavailableError,
            PrometheusQueryError,
            http=http,
            timeout_seconds=timeout_seconds,
            retry_policy=retry_policy,
            breaker=breaker,
            sleep=sleep,
        )

    async def query_value(self, promql: str) -> float | None:
        """Instant query returning one number, or None when there is no (finite) sample."""
        data = await self.get_data("/api/v1/query", {"query": promql})
        return _first_value(data)

    async def query_range(
        self, promql: str, start: float, end: float, step_seconds: float
    ) -> tuple[tuple[float, float], ...]:
        """Range query returning the (unix time, value) points of one series; NaN points dropped."""
        params = {"query": promql, "start": start, "end": end, "step": step_seconds}
        data = await self.get_data("/api/v1/query_range", params)
        if data["resultType"] != "matrix" or not data["result"]:
            return ()
        points = ((float(t), float(v)) for t, v in data["result"][0]["values"])
        return tuple((t, v) for t, v in points if math.isfinite(v))


def _first_value(data: dict[str, Any]) -> float | None:
    """Read an instant-query result. Empty vector, NaN and Inf all mean "not observable"."""
    if data["resultType"] == "scalar":
        raw = data["result"][1]
    elif data["resultType"] == "vector" and data["result"]:
        raw = data["result"][0]["value"][1]
    else:
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


# DNS-label shape. Validating it prevents PromQL injection through the service identifier.
_SERVICE_ID = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")

# Business traffic only: health probes, scrapes and fault-injection calls are not user requests.
_TRAFFIC = 'path!~"/health.*|/metrics|/chaos.*"'

_MB = 1024 * 1024


def build_queries(service_id: str) -> dict[str, str]:
    """One aggregated series per query, so each answer is a single number for the service."""
    sel = f'service_id="{service_id}"'
    requests = f"http_requests_total{{{sel},{_TRAFFIC}}}"
    errors = f'http_requests_total{{{sel},{_TRAFFIC},status=~"5.."}}'
    buckets = f"http_request_duration_seconds_bucket{{{sel},{_TRAFFIC}}}"
    return {
        "memory_mb": f"sum(process_resident_memory_bytes{{{sel}}}) / {_MB}",
        "memory_slope_mb_min": (
            f"sum(deriv(process_resident_memory_bytes{{{sel}}}[5m])) * 60 / {_MB}"
        ),
        "cpu_cores": f"sum(rate(process_cpu_seconds_total{{{sel}}}[1m]))",
        "request_rate_rps": f"sum(rate({requests}[1m]))",
        # "or vector(0)": no 5xx series yet means zero errors. No traffic at all stays None.
        "error_rate": f"(sum(rate({errors}[5m])) or vector(0)) / sum(rate({requests}[5m]))",
        "latency_p95_ms": (f"histogram_quantile(0.95, sum by (le) (rate({buckets}[5m]))) * 1000"),
    }


# Prometheus duration such as 5m or 1h; validated because it is interpolated into PromQL.
_WINDOW = re.compile(r"^[1-9][0-9]*[smhd]$")
# Histogram bucket bound as printed by prometheus_client (e.g. "0.5", "1.0").
_BUCKET_LE = re.compile(r"^[0-9]+\.[0-9]+$")


def build_sli_queries(service_id: str, window: str, latency_le: str) -> dict[str, str]:
    """Bad-event ratios over one window, for the availability and latency SLOs.

    Availability: a request is bad when it returns 5xx.
    Latency: a request is bad when it takes longer than the `latency_le` histogram bucket.
    """
    sel = f'service_id="{service_id}"'
    requests = f"http_requests_total{{{sel},{_TRAFFIC}}}"
    errors = f'http_requests_total{{{sel},{_TRAFFIC},status=~"5.."}}'
    fast = f'http_request_duration_seconds_bucket{{{sel},{_TRAFFIC},le="{latency_le}"}}'
    observed = f"http_request_duration_seconds_count{{{sel},{_TRAFFIC}}}"
    return {
        "request_rate_rps": f"sum(rate({requests}[{window}]))",
        "error_ratio": (
            f"(sum(rate({errors}[{window}])) or vector(0)) / sum(rate({requests}[{window}]))"
        ),
        "slow_ratio": f"1 - sum(rate({fast}[{window}])) / sum(rate({observed}[{window}]))",
    }


def build_series_queries(service_id: str) -> dict[str, str]:
    """Signals tracked over time by the statistical detector (raw levels, not ratios)."""
    sel = f'service_id="{service_id}"'
    buckets = f"http_request_duration_seconds_bucket{{{sel},{_TRAFFIC}}}"
    return {
        "memory_mb": f"sum(process_resident_memory_bytes{{{sel}}}) / {_MB}",
        "cpu_cores": f"sum(rate(process_cpu_seconds_total{{{sel}}}[1m]))",
        "latency_p95_ms": (f"histogram_quantile(0.95, sum by (le) (rate({buckets}[1m]))) * 1000"),
    }


class PrometheusCollector:
    def __init__(
        self,
        client: PrometheusClient,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._client = client
        self._clock = clock

    async def _ask_all(self, queries: dict[str, str]) -> tuple[dict[str, float | None], int]:
        """Run queries concurrently. Unavailability becomes None; query errors are raised."""
        names = list(queries)
        answers = await asyncio.gather(
            *(self._client.query_value(queries[name]) for name in names),
            return_exceptions=True,
        )
        values: dict[str, float | None] = {}
        unavailable = 0
        for name, answer in zip(names, answers, strict=True):
            if isinstance(answer, PrometheusUnavailableError | CircuitOpenError):
                unavailable += 1
                answer = None
            elif isinstance(answer, BaseException):
                raise answer  # PrometheusQueryError or a bug: never hide it as missing data
            values[name] = answer
        return values, unavailable

    async def collect_sli(
        self, service_id: str, windows: Iterable[str], latency_le: str
    ) -> dict[str, SLIWindow]:
        """Bad-event ratios for each window, e.g. {"5m": SLIWindow(...), "1h": ...}."""
        check_service_id(service_id)
        if not _BUCKET_LE.fullmatch(latency_le):
            raise ValueError(f"Invalid latency bucket: {latency_le!r}")
        windows = sorted(set(windows))
        for window in windows:
            if not _WINDOW.fullmatch(window):
                raise ValueError(f"Invalid window: {window!r}")
        per_window = await asyncio.gather(
            *(self._ask_all(build_sli_queries(service_id, w, latency_le)) for w in windows)
        )
        return {
            window: SLIWindow(
                window=window,
                request_rate_rps=values["request_rate_rps"],
                error_ratio=values["error_ratio"],
                slow_ratio=values["slow_ratio"],
                source_available=unavailable < len(values),
            )
            for window, (values, unavailable) in zip(windows, per_window, strict=True)
        }

    async def collect_series(
        self, service_id: str, lookback_seconds: float, step_seconds: float
    ) -> dict[str, Series]:
        """History of each tracked signal over the last `lookback_seconds`.

        A signal whose source is unavailable comes back with no points (unknown), so one
        failing query never hides the other signals.
        """
        check_service_id(service_id)
        end = self._clock().timestamp()
        queries = build_series_queries(service_id)
        names = list(queries)
        answers = await asyncio.gather(
            *(
                self._client.query_range(queries[n], end - lookback_seconds, end, step_seconds)
                for n in names
            ),
            return_exceptions=True,
        )
        series: dict[str, Series] = {}
        for name, answer in zip(names, answers, strict=True):
            if isinstance(answer, PrometheusUnavailableError | CircuitOpenError):
                series[name] = Series(signal=name, points=(), source_available=False)
            elif isinstance(answer, BaseException):
                raise answer
            else:
                series[name] = Series(signal=name, points=answer, source_available=True)
        return series

    async def collect(self, service_id: str) -> MetricSnapshot:
        check_service_id(service_id)
        values, unavailable = await self._ask_all(build_queries(service_id))
        missing = [name for name, value in values.items() if value is None]

        return MetricSnapshot(
            service_id=service_id,
            collected_at=self._clock(),
            memory_mb=values["memory_mb"],
            memory_slope_mb_min=values["memory_slope_mb_min"],
            cpu_cores=values["cpu_cores"],
            request_rate_rps=values["request_rate_rps"],
            error_rate=values["error_rate"],
            latency_p95_ms=values["latency_p95_ms"],
            missing=tuple(missing),
            source_available=unavailable < len(values),
        )


def check_service_id(service_id: str) -> None:
    if not _SERVICE_ID.fullmatch(service_id):
        raise ValueError(f"Invalid service_id: {service_id!r}")
