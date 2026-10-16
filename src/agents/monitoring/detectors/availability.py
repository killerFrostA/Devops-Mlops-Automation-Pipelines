"""Layer 0: is the service there at all?

The other detectors read the service's signals. When the service crashes or stops, those
signals disappear and the other detectors can only say "insufficient data". This detector
turns that silence into a fact: the service has not answered once during `down_after_seconds`.

App mode: Prometheus `up` (every scrape of /metrics failed during the window).
Container mode: cAdvisor has not seen the container during the window.

DOWN is reported as a `readiness` symptom of severity HIGH (always alerted). A service that
Prometheus never scraped is UNKNOWN, not DOWN: a missing configuration is not an outage.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class AvailabilityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    # A few failed scrapes (15 s each) can be a hiccup; a full minute without one answer is not.
    down_after_seconds: int = Field(default=60, ge=30, le=3600)


class Availability(StrEnum):
    UP = "UP"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class AvailabilityFinding:
    status: Availability
    window_seconds: int


class AliveSource(Protocol):
    async def collect_alive(self, service_id: str, window_seconds: int) -> bool | None: ...


class AvailabilityDetector:
    def __init__(self, source: AliveSource, policy: AvailabilityPolicy | None = None) -> None:
        self._source = source
        self.policy = policy or AvailabilityPolicy()

    async def evaluate(self, service_id: str) -> list[AvailabilityFinding]:
        window = self.policy.down_after_seconds
        alive = await self._source.collect_alive(service_id, window)
        status = (
            Availability.UNKNOWN
            if alive is None
            else Availability.UP
            if alive
            else Availability.DOWN
        )
        return [AvailabilityFinding(status, window)]
