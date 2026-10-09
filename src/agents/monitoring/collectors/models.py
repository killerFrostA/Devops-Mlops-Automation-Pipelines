from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    """Typed observation of one service at one instant.

    A value is None when it could not be observed (missing series, no traffic, source down).
    None means "unknown", never zero: detectors must not treat missing data as healthy data.
    """

    service_id: str
    collected_at: datetime
    memory_mb: float | None
    memory_slope_mb_min: float | None
    cpu_cores: float | None
    request_rate_rps: float | None
    error_rate: float | None
    latency_p95_ms: float | None
    missing: tuple[str, ...]
    source_available: bool

    @property
    def complete(self) -> bool:
        return self.source_available and not self.missing


@dataclass(frozen=True, slots=True)
class Series:
    """Time series of one signal: (unix seconds, value) points, oldest first, gaps allowed."""

    signal: str
    points: tuple[tuple[float, float], ...]
    source_available: bool


@dataclass(frozen=True, slots=True)
class LogLine:
    timestamp: float  # unix seconds
    text: str


@dataclass(frozen=True, slots=True)
class SLIWindow:
    """Bad-event ratios (0..1) over one time window, for SLO burn-rate evaluation.

    None means unknown (no traffic or source unavailable), never "no bad events".
    """

    window: str
    request_rate_rps: float | None
    error_ratio: float | None
    slow_ratio: float | None
    source_available: bool
