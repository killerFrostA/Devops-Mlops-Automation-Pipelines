"""Read-only HTTP access to telemetry backends with timeout, retry and circuit breaker.

Prometheus and Loki share the same response shape ({"status": "success", "data": ...}), so
both collectors use this client and only differ in their queries and result parsing.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Self

import httpx

from src.platform.messaging.reliability import RetryPolicy
from src.platform.resilience import CircuitBreaker, retry_async


class SourceUnavailableError(RuntimeError):
    """Transport error, timeout or 5xx: transient, retried, and counted by the circuit breaker."""


class SourceQueryError(RuntimeError):
    """The backend rejected the query (4xx or error status): a caller bug, never retried."""


class GuardedClient:
    def __init__(
        self,
        base_url: str,
        unavailable_error: type[SourceUnavailableError] = SourceUnavailableError,
        query_error: type[SourceQueryError] = SourceQueryError,
        http: httpx.AsyncClient | None = None,
        timeout_seconds: float = 5.0,
        retry_policy: RetryPolicy | None = None,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._http = http or httpx.AsyncClient(base_url=base_url)
        self._unavailable = unavailable_error
        self._query_error = query_error
        self._timeout = timeout_seconds
        self._retry_policy = retry_policy or RetryPolicy()
        self._breaker = breaker or CircuitBreaker(trip_on=(unavailable_error,))
        self._sleep = sleep

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._http.aclose()

    async def get_data(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET `path` behind timeout, retry on transient errors, and the circuit breaker."""

        async def attempt() -> dict[str, Any]:
            return await self._request_once(path, params)

        async def with_retry() -> dict[str, Any]:
            return await retry_async(
                attempt, self._retry_policy, (self._unavailable,), sleep=self._sleep
            )

        return await self._breaker.call(with_retry)

    async def _request_once(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._http.get(path, params=params, timeout=self._timeout)
        except httpx.TransportError as error:  # includes connection errors and timeouts
            raise self._unavailable(f"Backend unreachable: {error}") from error
        if response.status_code >= 500:
            raise self._unavailable(f"Backend returned {response.status_code}")
        if response.status_code != 200:
            raise self._query_error(f"Query rejected ({response.status_code}): {params}")
        body: dict[str, Any] = response.json()
        if body.get("status") != "success":
            raise self._query_error(f"Query failed: {body.get('error', 'unknown error')}")
        data: dict[str, Any] = body["data"]
        return data
