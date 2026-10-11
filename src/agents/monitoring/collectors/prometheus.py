"""Prometheus collector: turns PromQL answers into a typed MetricSnapshot.

Read-only. Each query is bounded by a timeout, retried on transient failures and guarded by a
circuit breaker, so an unavailable Prometheus degrades into "unknown" values instead of a crash.

Queries are built from each service's profile (collectors/profile.py): the service's own
metrics in app mode, or cAdvisor container metrics in container mode. Nothing here is specific
to one service or project.
"""

import asyncio
import math
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

import httpx

from src.agents.monitoring.collectors.http import (
    GuardedClient,
    SourceQueryError,
    SourceUnavailableError,
)
from src.agents.monitoring.collectors.models import MetricSnapshot, Series, SLIWindow
from src.agents.monitoring.collectors.profile import MetricNames, Mode, ServiceProfile
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

_MB = 1024 * 1024

DEFAULT_PROFILE = ServiceProfile()


def _app_selectors(service_id: str, names: MetricNames) -> tuple[str, str, str]:
    """Label selectors of an instrumented service: all traffic, errors only, histogram."""
    sel = f'service_id="{service_id}"'
    if names.excluded_paths:
        # Business traffic only: health probes, scrapes and fault injection are not user requests.
        sel += f',{names.path_label}!~"{names.excluded_paths}"'
    return sel, f'{sel},{names.status_label}=~"5.."', sel


def _resource_queries(service_id: str, profile: ServiceProfile) -> tuple[str, str]:
    """Memory series (bytes) and CPU usage query (cores) of the service, in either mode."""
    if profile.mode is Mode.CONTAINER:
        # cAdvisor, from outside the container: no instrumentation needed in the service.
        # It measures every 30 s with its own timestamps, so a 1m rate often has a single
        # sample and no value: 2m always holds at least two.
        sel = f'name="{profile.container or service_id}"'
        return (
            f"container_memory_working_set_bytes{{{sel}}}",
            f"sum(rate(container_cpu_usage_seconds_total{{{sel}}}[2m]))",
        )
    sel = f'service_id="{service_id}"'
    return (
        f"{profile.metrics.memory_bytes}{{{sel}}}",
        f"sum(rate({profile.metrics.cpu_seconds}{{{sel}}}[1m]))",
    )


def build_queries(service_id: str, profile: ServiceProfile = DEFAULT_PROFILE) -> dict[str, str]:
    """One aggregated series per query, so each answer is a single number for the service.

    In container mode only memory and CPU exist: the request signals are left out.
    """
    memory, cpu = _resource_queries(service_id, profile)
    queries = {
        "memory_mb": f"sum({memory}) / {_MB}",
        "memory_slope_mb_min": f"sum(deriv({memory}[5m])) * 60 / {_MB}",
        "cpu_cores": cpu,
    }
    if profile.mode is Mode.CONTAINER:
        return queries
    names = profile.metrics
    sel, err, hist = _app_selectors(service_id, names)
    requests = f"{names.requests}{{{sel}}}"
    errors = f"{names.requests}{{{err}}}"
    buckets = f"{names.latency_histogram}_bucket{{{hist}}}"
    return queries | {
        "request_rate_rps": f"sum(rate({requests}[1m]))",
        # "or vector(0)": no 5xx series yet means zero errors. No traffic at all stays None.
        "error_rate": f"(sum(rate({errors}[5m])) or vector(0)) / sum(rate({requests}[5m]))",
        "latency_p95_ms": f"histogram_quantile(0.95, sum by (le) (rate({buckets}[5m]))) * 1000",
    }


# Prometheus duration such as 5m or 1h; validated because it is interpolated into PromQL.
_WINDOW = re.compile(r"^[1-9][0-9]*[smhd]$")
# Histogram bucket bound as printed by prometheus_client (e.g. "0.5", "1.0").
_BUCKET_LE = re.compile(r"^[0-9]+\.[0-9]+$")


def build_sli_queries(
    service_id: str, window: str, latency_le: str, profile: ServiceProfile = DEFAULT_PROFILE
) -> dict[str, str]:
    """Bad-event ratios over one window, for the availability and latency SLOs.

    Availability: a request is bad when it returns 5xx.
    Latency: a request is bad when it takes longer than the `latency_le` histogram bucket.
    Container mode has no request counter: no SLI query at all.
    """
    if profile.mode is Mode.CONTAINER:
        return {}
    names = profile.metrics
    sel, err, hist = _app_selectors(service_id, names)
    requests = f"{names.requests}{{{sel}}}"
    errors = f"{names.requests}{{{err}}}"
    fast = f'{names.latency_histogram}_bucket{{{hist},le="{latency_le}"}}'
    observed = f"{names.latency_histogram}_count{{{hist}}}"
    return {
        "request_rate_rps": f"sum(rate({requests}[{window}]))",
        "error_ratio": (
            f"(sum(rate({errors}[{window}])) or vector(0)) / sum(rate({requests}[{window}]))"
        ),
        "slow_ratio": f"1 - sum(rate({fast}[{window}])) / sum(rate({observed}[{window}]))",
    }


def build_series_queries(
    service_id: str, profile: ServiceProfile = DEFAULT_PROFILE
) -> dict[str, str]:
    """Signals tracked over time by the statistical detector (raw levels, not ratios)."""
    memory, cpu = _resource_queries(service_id, profile)
    queries = {"memory_mb": f"sum({memory}) / {_MB}", "cpu_cores": cpu}
    if profile.mode is Mode.CONTAINER:
        return queries
    names = profile.metrics
    _, _, hist = _app_selectors(service_id, names)
    buckets = f"{names.latency_histogram}_bucket{{{hist}}}"
    return queries | {
        "latency_p95_ms": f"histogram_quantile(0.95, sum by (le) (rate({buckets}[1m]))) * 1000"
    }


class PrometheusCollector:
    def __init__(
        self,
        client: PrometheusClient,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        profiles: Mapping[str, ServiceProfile] | None = None,
    ) -> None:
        self._client = client
        self._clock = clock
        self._profiles = dict(profiles or {})  # service_id -> how to measure it

    def profile(self, service_id: str) -> ServiceProfile:
        return self._profiles.get(service_id, DEFAULT_PROFILE)

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
        profile = self.profile(service_id)
        per_window = await asyncio.gather(
            *(self._ask_all(build_sli_queries(service_id, w, latency_le, profile)) for w in windows)
        )
        return {
            window: SLIWindow(
                window=window,
                request_rate_rps=values.get("request_rate_rps"),
                error_ratio=values.get("error_ratio"),
                slow_ratio=values.get("slow_ratio"),
                source_available=not values or unavailable < len(values),
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
        queries = build_series_queries(service_id, self.profile(service_id))
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
        values, unavailable = await self._ask_all(
            build_queries(service_id, self.profile(service_id))
        )
        missing = [name for name, value in values.items() if value is None]

        return MetricSnapshot(
            service_id=service_id,
            collected_at=self._clock(),
            memory_mb=values["memory_mb"],
            memory_slope_mb_min=values["memory_slope_mb_min"],
            cpu_cores=values["cpu_cores"],
            request_rate_rps=values.get("request_rate_rps"),
            error_rate=values.get("error_rate"),
            latency_p95_ms=values.get("latency_p95_ms"),
            missing=tuple(missing),
            source_available=unavailable < len(values),
        )


def check_service_id(service_id: str) -> None:
    if not _SERVICE_ID.fullmatch(service_id):
        raise ValueError(f"Invalid service_id: {service_id!r}")
