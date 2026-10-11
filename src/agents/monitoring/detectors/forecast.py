"""Layer 2 bis: forecast-based detection with Chronos, a pretrained time-series model (Amazon).

For each signal, the model reads the baseline window (the usual behaviour) and predicts the
recent window with an interval (e.g. 10 % - 90 % quantiles). The recent points are then
compared with what was predicted:
- ANOMALY when most recent points are above the predicted upper bound (`min_outside_share`,
  so a single spike is not enough) and the typical excess matters (`min_absolute_change`).

The model is used as is (zero-shot): nothing is trained on the service's data. It runs on CPU
in a worker thread so the loop stays responsive. Like the other detectors, findings describe
what was observed (higher than expected), never why.

The model is behind the `Forecaster` protocol: tests use a fake one, and torch/chronos are
only imported when the real model is used (optional extra `forecast`).
"""

import asyncio
import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.agents.monitoring.collectors.models import Series
from src.agents.monitoring.detectors.statistical import (
    FindingStatus,
    SeriesSource,
    StatisticalFinding,
)
from src.contracts.enums import Severity


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ForecastRule(_Config):
    signal: str
    category: str
    # Smallest excess over the predicted median that matters, in the signal's unit.
    min_absolute_change: float = Field(ge=0)


class ForecastPolicy(_Config):
    enabled: bool = False  # needs the optional extra: pip install -e ".[forecast]"
    model: str = "amazon/chronos-bolt-tiny"  # 9 M parameters, ~0.1 s per prediction on CPU
    threads: int = Field(default=2, ge=1)  # CPU threads given to the model
    baseline_seconds: float = Field(default=1800, gt=0)  # what the model reads
    recent_seconds: float = Field(default=300, gt=0)  # what it predicts and is checked
    step_seconds: float = Field(default=15, gt=0)
    min_coverage: float = Field(default=0.6, gt=0, le=1)
    low_quantile: float = Field(default=0.1, gt=0, lt=0.5)
    high_quantile: float = Field(default=0.9, gt=0.5, lt=1)
    min_outside_share: float = Field(default=0.6, gt=0, le=1)
    rules: tuple[ForecastRule, ...] = (
        ForecastRule(signal="memory_mb", category="memory_pressure", min_absolute_change=20),
        ForecastRule(signal="latency_p95_ms", category="latency", min_absolute_change=50),
        ForecastRule(signal="cpu_cores", category="cpu_pressure", min_absolute_change=0.1),
    )

    @model_validator(mode="after")
    def whole_steps(self) -> "ForecastPolicy":
        if self.recent_seconds < self.step_seconds:
            raise ValueError("recent_seconds must cover at least one step")
        return self

    @property
    def horizon(self) -> int:
        return int(self.recent_seconds // self.step_seconds)


class Forecaster(Protocol):
    def predict(
        self, context: Sequence[float], horizon: int, quantiles: Sequence[float]
    ) -> list[list[float]]:
        """One row per future step, one value per requested quantile (same order)."""
        ...


def _covered(points: Sequence[tuple[float, float]], seconds: float, policy: ForecastPolicy) -> bool:
    return len(points) >= max(3, policy.min_coverage * seconds / policy.step_seconds)


def evaluate(
    policy: ForecastPolicy, forecaster: Forecaster, series: dict[str, Series], now: float
) -> list[StatisticalFinding]:
    """Compare each signal's recent points with the model's forecast. `now` is unix seconds.

    Signals the service does not provide at all (absent from `series`) are skipped.
    """
    recent_start = now - policy.recent_seconds
    baseline_start = recent_start - policy.baseline_seconds
    quantiles = [policy.low_quantile, 0.5, policy.high_quantile]
    findings: list[StatisticalFinding] = []
    for rule in policy.rules:
        if rule.signal not in series:
            continue
        points = series[rule.signal].points
        context = [p for p in points if baseline_start <= p[0] < recent_start]
        recent = [p for p in points if recent_start <= p[0] <= now]
        if not (
            _covered(context, policy.baseline_seconds, policy)
            and _covered(recent, policy.recent_seconds, policy)
        ):
            findings.append(
                StatisticalFinding(
                    rule.signal, rule.category, "forecast", FindingStatus.INSUFFICIENT_DATA
                )
            )
            continue
        rows = forecaster.predict([v for _, v in context], policy.horizon, quantiles)
        last_context_time = context[-1][0]
        median_pred: list[float] = []
        high_pred: list[float] = []
        above = 0
        excess: list[float] = []
        for t, value in recent:
            # Forecast step matching this point (step 0 = first step after the context).
            step = round((t - last_context_time) / policy.step_seconds) - 1
            _, median, high = rows[min(max(step, 0), len(rows) - 1)]
            median_pred.append(median)
            high_pred.append(high)
            above += value > high
            excess.append(value - median)
        share = above / len(recent)
        current = statistics.median(v for _, v in recent)
        expected = statistics.median(median_pred)
        typical_excess = statistics.median(excess)
        anomalous = share >= policy.min_outside_share and typical_excess >= rule.min_absolute_change
        findings.append(
            StatisticalFinding(
                signal=rule.signal,
                category=rule.category,
                method="forecast",
                status=FindingStatus.ANOMALY if anomalous else FindingStatus.NORMAL,
                current=current,
                baseline_median=expected,
                relative_change=(current - expected) / abs(expected) if expected else None,
                severity=Severity.MEDIUM if anomalous else None,
                expected_high=statistics.median(high_pred),
                outside_share=share,
            )
        )
    return findings


class ForecastDetector:
    def __init__(
        self,
        source: SeriesSource,
        forecaster: Forecaster,
        policy: ForecastPolicy | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._source = source
        self._forecaster = forecaster
        self.policy = policy or ForecastPolicy()
        self._clock = clock

    async def evaluate(self, service_id: str) -> list[StatisticalFinding]:
        lookback = self.policy.baseline_seconds + self.policy.recent_seconds
        series = await self._source.collect_series(service_id, lookback, self.policy.step_seconds)
        now = self._clock()
        # CPU-bound model inference: off the event loop, so other services keep being watched.
        return await asyncio.to_thread(evaluate, self.policy, self._forecaster, series, now)


class ChronosForecaster:
    """Amazon Chronos(-Bolt) on CPU, loaded on first use (a few seconds, then cached)."""

    def __init__(self, model: str, threads: int = 2) -> None:
        self._model = model
        self._threads = threads
        self._pipeline: Any = None

    def warm_up(self) -> None:
        """Load the model now (import + weights take seconds) rather than in the first cycle."""
        self._load()

    def _load(self) -> Any:
        if self._pipeline is None:
            import torch  # optional dependency, imported only when the model is used
            from chronos import BaseChronosPipeline

            torch.set_num_threads(self._threads)
            self._pipeline = BaseChronosPipeline.from_pretrained(
                self._model, device_map="cpu", torch_dtype=torch.float32
            )
        return self._pipeline

    def predict(
        self, context: Sequence[float], horizon: int, quantiles: Sequence[float]
    ) -> list[list[float]]:
        import torch

        pipeline = self._load()
        with torch.inference_mode():
            predicted, _ = pipeline.predict_quantiles(
                torch.tensor(list(context), dtype=torch.float32),
                prediction_length=horizon,
                quantile_levels=list(quantiles),
            )
        rows: list[list[float]] = predicted[0].tolist()
        return rows
