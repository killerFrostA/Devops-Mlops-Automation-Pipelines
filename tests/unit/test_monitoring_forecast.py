import asyncio
import os
import random
import statistics
from collections.abc import Callable, Sequence

import pytest

from src.agents.monitoring.collectors.models import Series
from src.agents.monitoring.config import MonitoringConfig, WatchedService
from src.agents.monitoring.detectors.forecast import (
    ChronosForecaster,
    ForecastDetector,
    ForecastPolicy,
    evaluate,
)
from src.agents.monitoring.detectors.statistical import FindingStatus, StatisticalFinding
from src.agents.monitoring.grouping import InMemoryIncidentStore, SymptomGrouper, from_statistical
from src.agents.monitoring.loop import ConsoleReporter, MonitoringLoop

POLICY = ForecastPolicy(enabled=True)
NOW = 1_800_000_000.0
START = NOW - POLICY.baseline_seconds - POLICY.recent_seconds


class UsualForecaster:
    """Predicts that the usual level continues: median of the context, +/- its spread."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def predict(
        self, context: Sequence[float], horizon: int, quantiles: Sequence[float]
    ) -> list[list[float]]:
        self.calls.append(len(context))
        median = statistics.median(context)
        spread = max(statistics.pstdev(context) * 1.3, 1e-6)
        return [[median - spread, median, median + spread] for _ in range(horizon)]


def series(signal: str, value_at: Callable[[float], float], seed: int = 1) -> Series:
    rng = random.Random(seed)
    steps = int((NOW - START) // POLICY.step_seconds)
    points = tuple(
        (
            START + i * POLICY.step_seconds,
            value_at(START + i * POLICY.step_seconds) + rng.gauss(0, 1),
        )
        for i in range(steps + 1)
    )
    return Series(signal=signal, points=points, source_available=True)


def recent(t: float) -> float:
    """Minutes elapsed in the recent window, 0 before it."""
    return max(0.0, (t - (NOW - POLICY.recent_seconds)) / 60)


def find(findings: list[StatisticalFinding], signal: str) -> StatisticalFinding:
    return next(f for f in findings if f.signal == signal)


def run(data: dict[str, Series]) -> list[StatisticalFinding]:
    return evaluate(POLICY, UsualForecaster(), data, NOW)


def test_usual_behaviour_is_normal() -> None:
    finding = find(run({"memory_mb": series("memory_mb", lambda _: 120.0)}), "memory_mb")
    assert finding.status is FindingStatus.NORMAL
    assert finding.method == "forecast"
    assert finding.outside_share is not None and finding.outside_share < 0.6


def test_leak_above_the_forecast_is_an_anomaly() -> None:
    leak = series("memory_mb", lambda t: 120.0 + 30 * recent(t))  # +30 MB/min in the window
    finding = find(run({"memory_mb": leak}), "memory_mb")
    assert finding.status is FindingStatus.ANOMALY
    assert finding.category == "memory_pressure"
    assert finding.outside_share is not None and finding.outside_share >= 0.6
    assert finding.baseline_median == pytest.approx(120, abs=2)  # what was expected


def test_a_single_spike_is_not_enough() -> None:
    spike_at = NOW - 60
    data = series("latency_p95_ms", lambda t: 900.0 if abs(t - spike_at) < 1 else 90.0)
    assert find(run({"latency_p95_ms": data}), "latency_p95_ms").status is FindingStatus.NORMAL


def test_a_rise_too_small_to_matter_is_normal() -> None:
    # Idle service: 0.020-0.022 core usually, then 0.07. Far above the narrow predicted band,
    # but only +0.05 core, below min_absolute_change (0.1): not worth an alert.
    def cpu(t: float) -> float:
        return 0.07 if recent(t) > 0 else 0.02 + 0.002 * (int(t) // 15 % 2)

    steps = int((NOW - START) // POLICY.step_seconds)
    points = tuple((START + i * 15.0, cpu(START + i * 15.0)) for i in range(steps + 1))
    finding = find(run({"cpu_cores": Series("cpu_cores", points, True)}), "cpu_cores")
    assert finding.outside_share is not None and finding.outside_share > 0.9  # clearly outside...
    assert finding.status is FindingStatus.NORMAL  # ...but too small to matter


def test_missing_history_is_insufficient_and_absent_signals_are_skipped() -> None:
    short = Series("memory_mb", series("memory_mb", lambda _: 100.0).points[-10:], True)
    findings = run({"memory_mb": short})
    assert [(f.signal, f.status) for f in findings] == [
        ("memory_mb", FindingStatus.INSUFFICIENT_DATA)
    ]  # latency and CPU were not provided: not measured, so not reported


def test_the_model_reads_only_the_baseline_and_predicts_the_recent_window() -> None:
    forecaster = UsualForecaster()
    evaluate(POLICY, forecaster, {"memory_mb": series("memory_mb", lambda _: 100.0)}, NOW)
    expected_context = POLICY.baseline_seconds / POLICY.step_seconds
    assert forecaster.calls == [pytest.approx(expected_context, abs=1)]


class Source:
    def __init__(self, data: dict[str, Series]) -> None:
        self.data = data

    async def collect_series(
        self, service_id: str, lookback_seconds: float, step_seconds: float
    ) -> dict[str, Series]:
        return self.data


def test_chronos_findings_become_a_forecast_symptom_and_reach_the_report() -> None:
    leak = {"memory_mb": series("memory_mb", lambda t: 120.0 + 30 * recent(t))}
    detector = ForecastDetector(Source(leak), UsualForecaster(), POLICY, clock=lambda: NOW)
    symptoms = from_statistical("payment-service", "local", asyncio.run(detector.evaluate("x")))
    assert [(s.category, s.source) for s in symptoms] == [("memory_pressure", "forecast")]

    class NoFinding:
        async def evaluate(self, service_id: str) -> list[StatisticalFinding]:
            return []

    lines: list[str] = []
    loop = MonitoringLoop(
        config=MonitoringConfig(services=(WatchedService(service_id="payment-service"),)),
        slo=NoFinding(),  # type: ignore[arg-type]
        statistical=NoFinding(),
        logs=None,
        grouper=SymptomGrouper(InMemoryIncidentStore()),
        reporter=ConsoleReporter(lines.append, verbose=True),
        forecast=detector,
    )
    cycle = asyncio.run(loop.run_cycle())
    assert cycle.services[0].forecast[0].status is FindingStatus.ANOMALY
    assert any(line.strip().startswith("Chronos      memory_mb") for line in lines)
    assert cycle.notifications[0].incident.sources == frozenset({"forecast"})


def test_disabled_by_default_and_rejects_a_window_shorter_than_a_step() -> None:
    assert not ForecastPolicy().enabled
    with pytest.raises(ValueError):
        ForecastPolicy(recent_seconds=10, step_seconds=15)


@pytest.mark.skipif(
    not os.environ.get("PIDS_RUN_CHRONOS"), reason="real model: set PIDS_RUN_CHRONOS=1 (~15 s)"
)
def test_real_chronos_model_on_cpu() -> None:
    pytest.importorskip("chronos")
    forecaster = ChronosForecaster("amazon/chronos-bolt-tiny")
    steady = {"memory_mb": series("memory_mb", lambda _: 120.0)}
    leak = {"memory_mb": series("memory_mb", lambda t: 120.0 + 30 * recent(t))}
    assert evaluate(POLICY, forecaster, steady, NOW)[0].status is FindingStatus.NORMAL
    assert evaluate(POLICY, forecaster, leak, NOW)[0].status is FindingStatus.ANOMALY
