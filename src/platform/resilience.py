"""Executable retry and circuit-breaker behaviour for calls to external tools.

`RetryPolicy` (messaging/reliability.py) only describes the budget; this module applies it.
Clock, sleep and randomness are injectable so the behaviour is testable without real waiting.
"""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum

from src.platform.messaging.reliability import RetryPolicy


class CircuitOpenError(RuntimeError):
    """The dependency failed repeatedly; calls are refused until the reset timeout elapses."""


class CircuitState(StrEnum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Stops calling a failing dependency, then lets one trial call through after a pause.

    CLOSED -> OPEN after `failure_threshold` consecutive failures.
    OPEN -> HALF_OPEN once `reset_timeout_seconds` have elapsed.
    HALF_OPEN -> CLOSED on success, back to OPEN on failure.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_seconds: float = 30.0,
        trip_on: tuple[type[BaseException], ...] = (Exception,),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1 or reset_timeout_seconds <= 0:
            raise ValueError("failure_threshold must be >= 1 and reset_timeout_seconds > 0")
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout_seconds
        # Only dependency failures trip the circuit; a caller's own bad request must not.
        self._trip_on = trip_on
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        if self._opened_at is None:
            return CircuitState.CLOSED
        if self._clock() - self._opened_at >= self._reset_timeout:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN

    async def call[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        if self.state is CircuitState.OPEN:
            raise CircuitOpenError("circuit open: dependency recently failed repeatedly")
        try:
            result = await operation()
        except self._trip_on:
            self._record_failure()
            raise
        self._failures = 0
        self._opened_at = None
        return result

    def _record_failure(self) -> None:
        if self.state is CircuitState.HALF_OPEN:
            self._opened_at = self._clock()  # trial call failed: open again for a full period
            return
        self._failures += 1
        if self._failures >= self._failure_threshold:
            self._opened_at = self._clock()


def backoff_delay(attempt: int, policy: RetryPolicy, rng: Callable[[], float]) -> float:
    """Exponential backoff with full jitter: a random delay in [0, min(max, initial * 2^n)]."""
    ceiling = min(policy.max_backoff_seconds, policy.initial_backoff_seconds * 2.0 ** (attempt - 1))
    return ceiling * rng()


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    retry_on: tuple[type[BaseException], ...],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: Callable[[], float] = random.random,
) -> T:
    """Run `operation`, retrying only `retry_on` errors, at most `policy.max_attempts` times."""
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await operation()
        except retry_on:
            if attempt == policy.max_attempts:
                raise
            await sleep(backoff_delay(attempt, policy, rng))
    raise AssertionError("unreachable: max_attempts >= 1")
