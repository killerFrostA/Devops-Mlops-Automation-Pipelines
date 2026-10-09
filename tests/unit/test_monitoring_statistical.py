import asyncio
import random
from collections.abc import Callable

import httpx
import pytest

from src.agents.monitoring.collectors.models import Series
from src.agents.monitoring.collectors.prometheus import PrometheusClient, PrometheusCollector
from src.agents.monitoring.detectors.statistical import (
    FindingStatus,
    StatisticalDetector,
    StatisticalFinding,
    StatisticalPolicy,
    evaluate,
    linear_fit,
    robust_z,
)
from src.contracts.enums import Severity

NOW = 1_000_000.0
STEP = 15.0
POLICY = StatisticalPolicy()  # 30 min baseline, then 5 min recent window
START = NOW - POLICY.baseline_seconds - POLICY.recent_seconds


def series(signal: str, value_at: Callable[[float], float], seed: int = 1) -> Series:
    """Points every 15 s over baseline + recent window; value_at(t) gets seconds since start."""
    rng = random.Random(seed)
    points = []
    t = START
    while t <= NOW:
        points.append((t, value_at(t - START) * (1 + rng.uniform(-0.03, 0.03))))
        t += STEP
    return Series(signal=signal, points=tuple(points), source_available=True)


def healthy() -> dict[str, Series]:
    return {
        "memory_mb": series("memory_mb", lambda _: 50.0),
        "latency_p95_ms": series("latency_p95_ms", lambda _: 90.0),
        "cpu_cores": series("cpu_cores", lambda _: 0.02),
    }


def find(findings: list[StatisticalFinding], signal: str) -> StatisticalFinding:
    return next(f for f in findings if f.signal == signal)


def after_baseline(seconds: float) -> bool:
    return seconds >= POLICY.baseline_seconds


def test_healthy_service_has_no_anomaly() -> None:
    findings = evaluate(POLICY, healthy(), NOW)
    assert all(f.status is FindingStatus.NORMAL for f in findings)


def test_memory_growth_is_detected_with_its_slope() -> None:
    data = healthy()
    # +30 MB/min during the recent window only, as injected by /chaos/memory?mb_per_min=30.
    data["memory_mb"] = series(
        "memory_mb",
        lambda s: 50 + (30 * (s - POLICY.baseline_seconds) / 60 if after_baseline(s) else 0),
    )
    memory = find(evaluate(POLICY, data, NOW), "memory_mb")
    assert memory.status is FindingStatus.ANOMALY
    assert memory.category == "memory_pressure"
    assert memory.severity is Severity.MEDIUM
    assert memory.slope_per_minute == pytest.approx(30, rel=0.15)


def test_latency_level_shift_below_slo_threshold_is_detected() -> None:
    data = healthy()
    data["latency_p95_ms"] = series(
        "latency_p95_ms", lambda s: 400.0 if after_baseline(s) else 90.0
    )
    latency = find(evaluate(POLICY, data, NOW), "latency_p95_ms")
    assert latency.status is FindingStatus.ANOMALY
    assert latency.baseline_median == pytest.approx(90, rel=0.05)
    assert latency.current == pytest.approx(400, rel=0.05)


def test_single_spike_is_not_an_anomaly() -> None:
    data = healthy()
    spiky = list(data["latency_p95_ms"].points)
    spiky[-1] = (spiky[-1][0], 2000.0)  # one extreme point, the most recent one
    data["latency_p95_ms"] = Series("latency_p95_ms", tuple(spiky), True)
    assert find(evaluate(POLICY, data, NOW), "latency_p95_ms").status is FindingStatus.NORMAL


def test_noisy_but_flat_memory_is_not_growth() -> None:
    rng = random.Random(7)
    data = healthy()
    data["memory_mb"] = series("memory_mb", lambda _: 50 + rng.uniform(-8, 8))
    assert find(evaluate(POLICY, data, NOW), "memory_mb").status is FindingStatus.NORMAL


def test_small_relative_change_on_flat_baseline_is_not_an_anomaly() -> None:
    """MAD = 0 on a perfectly flat baseline must not turn +10 % into an infinite z-score."""
    count = int((NOW - START) / STEP) + 1  # baseline + recent windows, every 15 s
    flat = Series("cpu_cores", tuple((START + i * STEP, 0.02) for i in range(count)), True)
    raised = tuple((t, 0.022 if t >= NOW - POLICY.recent_seconds else v) for t, v in flat.points)
    data = {**healthy(), "cpu_cores": Series("cpu_cores", raised, True)}
    assert find(evaluate(POLICY, data, NOW), "cpu_cores").status is FindingStatus.NORMAL


@pytest.mark.parametrize(
    "data",
    [
        {},  # signal never collected
        {"latency_p95_ms": Series("latency_p95_ms", (), False)},  # source unavailable
        # no traffic during the recent window: p95 is undefined, recent points are missing
        {
            "latency_p95_ms": Series(
                "latency_p95_ms", series("latency_p95_ms", lambda _: 90.0).points[:100], True
            )
        },
    ],
)
def test_missing_data_is_insufficient_not_normal(data: dict[str, Series]) -> None:
    latency = find(evaluate(POLICY, data, NOW), "latency_p95_ms")
    assert latency.status is FindingStatus.INSUFFICIENT_DATA


def test_linear_fit_and_robust_z() -> None:
    slope, r2 = linear_fit([(0, 10.0), (60, 40.0), (120, 70.0)])
    assert slope == pytest.approx(30)
    assert r2 == pytest.approx(1)
    assert robust_z(100, [10, 11, 9, 10, 10], 0.05) > 4


def test_detector_reads_history_from_prometheus_range_queries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/query_range"
        query = request.url.params["query"]
        start, end = float(request.url.params["start"]), float(request.url.params["end"])
        values = []
        t = start
        while t <= end:
            if "memory" in query:
                value = 50 + max(0.0, (t - (end - 300)) / 60 * 30)  # grows in the last 5 min
            elif "histogram" in query:
                value = 90.0
            else:
                value = 0.02
            values.append([t, str(value)])
            t += 15
        result = [{"metric": {}, "values": values}]
        return httpx.Response(
            200, json={"status": "success", "data": {"resultType": "matrix", "result": result}}
        )

    async def run() -> list[StatisticalFinding]:
        http = httpx.AsyncClient(base_url="http://prom", transport=httpx.MockTransport(handler))
        async with PrometheusClient("http://prom", http=http) as client:
            collector = PrometheusCollector(client)
            return await StatisticalDetector(collector).evaluate("payment-service")

    findings = asyncio.run(run())
    assert find(findings, "memory_mb").status is FindingStatus.ANOMALY
    assert find(findings, "latency_p95_ms").status is FindingStatus.NORMAL
