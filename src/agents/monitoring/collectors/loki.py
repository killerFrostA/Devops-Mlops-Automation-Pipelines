"""Loki collector: reads recent log lines of one service. Read-only.

Lines are labelled by Promtail with `service_id` (from the container label of the same name).
"""

import asyncio
from collections.abc import Awaitable, Callable

import httpx

from src.agents.monitoring.collectors.http import (
    GuardedClient,
    SourceQueryError,
    SourceUnavailableError,
)
from src.agents.monitoring.collectors.models import LogLine
from src.agents.monitoring.collectors.prometheus import check_service_id
from src.platform.messaging.reliability import RetryPolicy
from src.platform.resilience import CircuitBreaker

_NANOS = 1_000_000_000


class LokiUnavailableError(SourceUnavailableError):
    """Loki unreachable, timed out or 5xx: retried and counted by the circuit breaker."""


class LokiQueryError(SourceQueryError):
    """Loki rejected the LogQL query: a caller bug, never retried."""


class LokiClient(GuardedClient):
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
            LokiUnavailableError,
            LokiQueryError,
            http=http,
            timeout_seconds=timeout_seconds,
            retry_policy=retry_policy,
            breaker=breaker,
            sleep=sleep,
        )

    async def query_lines(
        self, service_id: str, start: float, end: float, limit: int = 5000
    ) -> tuple[LogLine, ...]:
        """Log lines of `service_id` between two unix times, oldest first, at most `limit`."""
        check_service_id(service_id)  # the id is interpolated into LogQL
        params = {
            "query": f'{{service_id="{service_id}"}}',
            "start": int(start * _NANOS),
            "end": int(end * _NANOS),
            "limit": limit,
            "direction": "forward",
        }
        data = await self.get_data("/loki/api/v1/query_range", params)
        lines = [
            LogLine(timestamp=int(ns) / _NANOS, text=text)
            for stream in data.get("result", [])
            for ns, text in stream["values"]
        ]
        return tuple(sorted(lines, key=lambda line: line.timestamp))
