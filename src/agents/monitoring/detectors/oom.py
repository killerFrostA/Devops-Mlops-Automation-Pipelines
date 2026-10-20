"""Layer 0 bis: was the service killed for lack of memory (OOM kill)?

Two observed facts, not guesses, are read over the window:
- cAdvisor's kernel OOM kill counter (`container_oom_events_total`). Reliable on a Linux host,
  but it stays at 0 under Docker Desktop / WSL2 (verified live: container killed, counter 0);
- a restart of the service (its process start time changed) while its memory had reached the
  container limit (peak working set / limit >= `memory_limit_ratio`). Verified live on WSL2.

Either one is an OOM kill: a `memory_oom` symptom of severity CRITICAL (P0), always alerted.
A restart WITHOUT memory at the limit is still an observed event (crash, unplanned restart):
reported as `readiness` HIGH, the cause being left to the Diagnosis agent.

A memory leak is usually seen earlier as memory_pressure (growth); this layer reports the
moment the limit was actually hit. The container is found by its `service_id` Docker label
(app mode) or by its name (container mode).
"""

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class OomPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    window_seconds: int = Field(default=300, ge=60, le=3600)
    # Peak memory / container limit from which a restart counts as an OOM kill.
    memory_limit_ratio: float = Field(default=0.9, gt=0, le=1)


@dataclass(frozen=True, slots=True)
class OomFinding:
    kills: float | None  # kernel OOM kills during the window; None = not measured
    window_seconds: int
    restarts: float | None = None  # process restarts during the window; None = not measured
    memory_peak_ratio: float | None = None  # peak memory / limit; None = no limit or unknown
    memory_limit_ratio: float = 0.9

    @property
    def at_limit(self) -> bool:
        return self.memory_peak_ratio is not None and (
            self.memory_peak_ratio >= self.memory_limit_ratio
        )

    @property
    def restarted(self) -> bool:
        return self.restarts is not None and self.restarts > 0

    @property
    def killed(self) -> bool:
        counted = self.kills is not None and self.kills > 0
        return counted or (self.restarted and self.at_limit)


class OomSource(Protocol):
    async def collect_oom_kills(self, service_id: str, window_seconds: int) -> float | None: ...

    async def collect_restarts(self, service_id: str, window_seconds: int) -> float | None: ...

    async def collect_memory_peak_ratio(
        self, service_id: str, window_seconds: int
    ) -> float | None: ...


class OomDetector:
    def __init__(self, source: OomSource, policy: OomPolicy | None = None) -> None:
        self._source = source
        self.policy = policy or OomPolicy()

    async def evaluate(self, service_id: str) -> list[OomFinding]:
        window = self.policy.window_seconds
        kills = await self._source.collect_oom_kills(service_id, window)
        restarts = await self._source.collect_restarts(service_id, window)
        # The memory peak only matters when the service restarted or was killed.
        ratio = None
        if (restarts or 0) > 0 or (kills or 0) > 0:
            ratio = await self._source.collect_memory_peak_ratio(service_id, window)
        return [OomFinding(kills, window, restarts, ratio, self.policy.memory_limit_ratio)]
