import asyncio
from collections.abc import Awaitable, Callable

import pytest

from src.platform.messaging.reliability import RetryPolicy
from src.platform.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    backoff_delay,
    retry_async,
)


class Transient(Exception):
    pass


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def no_sleep(_: float) -> None:
    return None


def flaky(failures: int) -> tuple[list[int], Callable[[], Awaitable[str]]]:
    calls: list[int] = []

    async def operation() -> str:
        calls.append(1)
        if len(calls) <= failures:
            raise Transient
        return "ok"

    return calls, operation


def test_retry_succeeds_after_transient_failures() -> None:
    calls, operation = flaky(failures=2)
    result = asyncio.run(
        retry_async(operation, RetryPolicy(max_attempts=3), (Transient,), sleep=no_sleep)
    )
    assert result == "ok"
    assert len(calls) == 3


def test_retry_gives_up_after_max_attempts() -> None:
    calls, operation = flaky(failures=10)
    with pytest.raises(Transient):
        asyncio.run(
            retry_async(operation, RetryPolicy(max_attempts=3), (Transient,), sleep=no_sleep)
        )
    assert len(calls) == 3


def test_retry_does_not_retry_other_errors() -> None:
    calls: list[int] = []

    async def operation() -> None:
        calls.append(1)
        raise ValueError("caller bug")

    with pytest.raises(ValueError):
        asyncio.run(retry_async(operation, RetryPolicy(), (Transient,), sleep=no_sleep))
    assert len(calls) == 1


def test_backoff_is_exponential_capped_and_jittered() -> None:
    policy = RetryPolicy(initial_backoff_seconds=1, max_backoff_seconds=5)
    assert [backoff_delay(n, policy, rng=lambda: 1.0) for n in (1, 2, 3, 4)] == [1, 2, 4, 5]
    assert backoff_delay(3, policy, rng=lambda: 0.5) == 2


def test_circuit_opens_then_half_opens_then_closes() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout_seconds=30, clock=clock)

    async def fail() -> None:
        raise Transient

    async def succeed() -> str:
        return "ok"

    for _ in range(2):
        with pytest.raises(Transient):
            asyncio.run(breaker.call(fail))
    assert breaker.state is CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        asyncio.run(breaker.call(succeed))

    clock.now = 30
    assert breaker.state is CircuitState.HALF_OPEN
    assert asyncio.run(breaker.call(succeed)) == "ok"
    assert breaker.state is CircuitState.CLOSED


def test_failed_trial_call_reopens_circuit() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=10, clock=clock)

    async def fail() -> None:
        raise Transient

    with pytest.raises(Transient):
        asyncio.run(breaker.call(fail))
    clock.now = 10
    with pytest.raises(Transient):
        asyncio.run(breaker.call(fail))
    assert breaker.state is CircuitState.OPEN


def test_errors_outside_trip_on_do_not_open_circuit() -> None:
    breaker = CircuitBreaker(failure_threshold=1, trip_on=(Transient,))

    async def bad_request() -> None:
        raise ValueError("caller bug")

    with pytest.raises(ValueError):
        asyncio.run(breaker.call(bad_request))
    assert breaker.state is CircuitState.CLOSED
