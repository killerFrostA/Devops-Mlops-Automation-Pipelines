"""Loki collector: reads recent log lines of one service. Read-only.

Promtail labels every container's lines with `container` (its Docker name), and with
`service_id` when the container declares that label. A service is read by `service_id`, or by
its container name when it is watched in container mode (no label needed on the container).
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping

import httpx

from src.agents.monitoring.collectors.http import (
    GuardedClient,
    SourceQueryError,
    SourceUnavailableError,
)
from src.agents.monitoring.collectors.models import LogLine
from src.agents.monitoring.collectors.profile import check_container
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
        containers: Mapping[str, str] | None = None,
    ) -> None:
        # service_id -> Docker container name, for services watched in container mode.
        self._containers = dict(containers or {})
        for name in self._containers.values():
            check_container(name)
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
        self,
        service_id: str,
        start: float,
        end: float,
        limit: int = 5000,
        line_regex: str | None = None,
    ) -> tuple[LogLine, ...]:
        """Log lines of `service_id` between two unix times, oldest first, at most `limit`.

        `line_regex` filters lines inside Loki (LogQL `|~`), so only matching lines travel:
        without it, a busy service's normal lines can fill `limit` and hide the rest.
        """
        check_service_id(service_id)  # the id is interpolated into LogQL
        container = self._containers.get(service_id)
        query = f'{{container="{container}"}}' if container else f'{{service_id="{service_id}"}}'
        if line_regex is not None:
            if "`" in line_regex:
                raise ValueError("line_regex must not contain a backtick")
            query += f" |~ `{line_regex}`"  # backticks: raw string, no escaping needed
        params = {
            "query": query,
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
