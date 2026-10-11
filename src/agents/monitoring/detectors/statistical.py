"""Layer 2: adaptive statistical detection, for anomalies that do not (yet) break an SLO.

Each signal is compared with its own recent past rather than a fixed threshold:
- growth: a sustained, regular increase (least-squares slope + R^2), e.g. memory climbing;
- level shift: the recent median sits far from the baseline median, measured in robust
  units (median absolute deviation), e.g. latency jumping from 90 ms to 400 ms.

Medians are used instead of means so that one isolated spike cannot trigger or mask an
anomaly: a shift must persist over more than half of the recent window to count.

Baseline = the window just before the recent window (trailing baseline). A seasonal baseline
(same hour on previous days) needs days of history and is a later extension.

Findings describe what was observed, never why.
"""

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.agents.monitoring.collectors.models import Series
from src.contracts.enums import Severity

# Scales the MAD so it estimates the standard deviation of normally distributed data.
MAD_TO_SIGMA = 1.4826


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GrowthRule(_Config):
    signal: str
    category: str
    min_slope_per_minute: float = Field(gt=0)
    min_r2: float = Field(default=0.8, ge=0, le=1)  # how regular the increase must be


class LevelShiftRule(_Config):
    signal: str
    category: str
    min_robust_z: float = Field(default=4.0, gt=0)
    min_relative_change: float = Field(gt=0)  # e.g. 0.5 = at least +50 % above baseline
    # Smallest change that matters, in the signal's unit. Without it, a baseline near zero
    # (an idle service) turns a negligible rise such as 0 -> 0.02 CPU core into an "anomaly".
    min_absolute_change: float = Field(default=0, ge=0)


class StatisticalPolicy(_Config):
    """Starting values; to be calibrated on the test bench (false alerts vs detection delay)."""

    baseline_seconds: float = Field(default=1800, gt=0)
    recent_seconds: float = Field(default=300, gt=0)
    step_seconds: float = Field(default=15, gt=0)
    min_coverage: float = Field(default=0.6, gt=0, le=1)  # share of expected points present
    # Floor of the robust scale, relative to the baseline median: a perfectly flat baseline
    # (MAD = 0) must not turn a tiny change into an infinite z-score.
    min_scale_fraction: float = Field(default=0.05, gt=0)
    growth: tuple[GrowthRule, ...] = (
        GrowthRule(signal="memory_mb", category="memory_pressure", min_slope_per_minute=5.0),
    )
    level_shifts: tuple[LevelShiftRule, ...] = (
        LevelShiftRule(
            signal="latency_p95_ms",
            category="latency",
            min_relative_change=0.5,
            min_absolute_change=50,  # ms
        ),
        LevelShiftRule(
            signal="cpu_cores",
            category="cpu_pressure",
            min_relative_change=1.0,
            min_absolute_change=0.1,  # cores
        ),
    )


class FindingStatus(StrEnum):
    ANOMALY = "ANOMALY"
    NORMAL = "NORMAL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True, slots=True)
class StatisticalFinding:
    signal: str
    category: str
    method: str  # "growth", "level_shift" or "forecast" (detectors/forecast.py)
    status: FindingStatus
    current: float | None = None  # recent median
    baseline_median: float | None = None  # forecast: median of the predicted values
    robust_z: float | None = None
    relative_change: float | None = None
    slope_per_minute: float | None = None
    r2: float | None = None
    severity: Severity | None = None  # set only on ANOMALY
    expected_high: float | None = None  # forecast: median of the predicted upper bound
    outside_share: float | None = None  # forecast: share of recent points above that bound


def linear_fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares slope (value units per minute) and R^2 of the points."""
    xs = [t / 60 for t, _ in points]
    ys = [v for _, v in points]
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    if sxx == 0:
        return 0.0, 0.0
    slope = sxy / sxx
    r2 = 0.0 if syy == 0 else (sxy * sxy) / (sxx * syy)
    return slope, r2


def robust_z(value: float, baseline: list[float], min_scale_fraction: float) -> float:
    """Distance of `value` from the baseline median, in robust standard deviations."""
    median = statistics.median(baseline)
    mad = statistics.median(abs(b - median) for b in baseline)
    scale = max(MAD_TO_SIGMA * mad, abs(median) * min_scale_fraction, 1e-9)
    return (value - median) / scale


def _covered(points: list[tuple[float, float]], seconds: float, policy: StatisticalPolicy) -> bool:
    expected = seconds / policy.step_seconds
    return len(points) >= max(3, policy.min_coverage * expected)


def evaluate(
    policy: StatisticalPolicy, series: dict[str, Series], now: float
) -> list[StatisticalFinding]:
    """Pure decision over already collected series. `now` is unix seconds."""
    recent_start = now - policy.recent_seconds
    baseline_start = recent_start - policy.baseline_seconds

    def split(signal: str) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        points = series[signal].points if signal in series else ()
        baseline = [p for p in points if baseline_start <= p[0] < recent_start]
        recent = [p for p in points if recent_start <= p[0] <= now]
        return baseline, recent

    findings: list[StatisticalFinding] = []
    for growth in policy.growth:
        _, recent = split(growth.signal)
        if not _covered(recent, policy.recent_seconds, policy):
            findings.append(
                StatisticalFinding(
                    growth.signal, growth.category, "growth", FindingStatus.INSUFFICIENT_DATA
                )
            )
            continue
        slope, r2 = linear_fit(recent)
        anomalous = slope >= growth.min_slope_per_minute and r2 >= growth.min_r2
        findings.append(
            StatisticalFinding(
                signal=growth.signal,
                category=growth.category,
                method="growth",
                status=FindingStatus.ANOMALY if anomalous else FindingStatus.NORMAL,
                current=statistics.median(v for _, v in recent),
                slope_per_minute=slope,
                r2=r2,
                severity=Severity.MEDIUM if anomalous else None,
            )
        )

    for shift in policy.level_shifts:
        baseline, recent = split(shift.signal)
        if not (
            _covered(baseline, policy.baseline_seconds, policy)
            and _covered(recent, policy.recent_seconds, policy)
        ):
            findings.append(
                StatisticalFinding(
                    shift.signal, shift.category, "level_shift", FindingStatus.INSUFFICIENT_DATA
                )
            )
            continue
        baseline_values = [v for _, v in baseline]
        current = statistics.median(v for _, v in recent)
        median = statistics.median(baseline_values)
        z = robust_z(current, baseline_values, policy.min_scale_fraction)
        relative = (current - median) / abs(median) if median else None
        anomalous = (
            z >= shift.min_robust_z
            and current - median >= shift.min_absolute_change
            and (relative is None or relative >= shift.min_relative_change)
        )
        findings.append(
            StatisticalFinding(
                signal=shift.signal,
                category=shift.category,
                method="level_shift",
                status=FindingStatus.ANOMALY if anomalous else FindingStatus.NORMAL,
                current=current,
                baseline_median=median,
                robust_z=z,
                relative_change=relative,
                severity=Severity.MEDIUM if anomalous else None,
            )
        )
    return findings


class SeriesSource(Protocol):
    async def collect_series(
        self, service_id: str, lookback_seconds: float, step_seconds: float
    ) -> dict[str, Series]: ...


class StatisticalDetector:
    def __init__(
        self,
        source: SeriesSource,
        policy: StatisticalPolicy | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._source = source
        self.policy = policy or StatisticalPolicy()
        self._clock = clock

    async def evaluate(self, service_id: str) -> list[StatisticalFinding]:
        lookback = self.policy.baseline_seconds + self.policy.recent_seconds
        series = await self._source.collect_series(service_id, lookback, self.policy.step_seconds)
        # Only the signals the service can provide: a container watched from outside has no
        # latency, which is "not measured" rather than "insufficient data".
        return [f for f in evaluate(self.policy, series, self._clock()) if f.signal in series]
