"""Independent recovery verification: did the action really fix the service?

When the Deployment & Recovery agent reports an action (`deployment.action.completed`), Agent 1
does not trust that report: it waits for the action to take effect, then samples the service's
own signals over an observation window and checks each one against its limit.

Per check (error rate, latency p95, memory growth):
- PASS: enough samples, most of them within the limit, and the last one too (still fine now);
- FAIL: enough samples, but too many over the limit or the last one over it;
- UNKNOWN: not enough samples (no traffic, source down, signal not measured for this service).

Status:
- NOT_RECOVERED as soon as one check fails;
- INCONCLUSIVE when a check behind the original anomaly is UNKNOWN (the problem that was
  reported cannot be shown to be gone), or when nothing could be measured;
- RECOVERED otherwise.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from src.agents.monitoring.decision import SignalReader
from src.agents.monitoring.events import build_recovery_event, new_id
from src.contracts.enums import Environment
from src.contracts.events import EventEnvelope
from src.contracts.payloads import ActionCompleted, RecoveryVerified
from src.contracts.tasks import TaskContext
from src.platform.ports import EventPublisher

CheckResult = Literal["PASS", "FAIL", "UNKNOWN"]

# Which check proves that a reported symptom category is gone.
CHECK_FOR_CATEGORY = {
    "error_rate": "error_rate",
    "latency": "latency_p95_ms",
    "memory_pressure": "memory_slope_mb_min",
}


class RecoveryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    settle_seconds: float = Field(default=30, ge=0)  # let the action take effect first
    window_seconds: float = Field(default=300, gt=0)  # observation window after that
    sample_seconds: float = Field(default=30, gt=0)
    min_samples_share: float = Field(default=0.5, gt=0, le=1)  # below: UNKNOWN
    pass_share: float = Field(default=0.8, gt=0, le=1)  # samples within the limit for PASS
    max_error_rate: float = Field(default=0.02, ge=0)
    max_latency_p95_ms: float = Field(default=500, gt=0)
    max_memory_slope_mb_min: float = Field(default=5, gt=0)

    @property
    def samples(self) -> int:
        return max(1, int(self.window_seconds // self.sample_seconds))

    @property
    def window(self) -> str:
        seconds = int(self.window_seconds)
        return f"{seconds // 60}m" if seconds % 60 == 0 else f"{seconds}s"

    def limits(self) -> dict[str, float]:
        return {
            "error_rate": self.max_error_rate,
            "latency_p95_ms": self.max_latency_p95_ms,
            "memory_slope_mb_min": self.max_memory_slope_mb_min,
        }


def judge(samples: list[dict[str, float | None]], policy: RecoveryPolicy) -> dict[str, CheckResult]:
    """PASS/FAIL/UNKNOWN per check from the samples taken during the window."""
    checks: dict[str, CheckResult] = {}
    for name, limit in policy.limits().items():
        known = [s[name] for s in samples if s.get(name) is not None]
        values = [v for v in known if v is not None]
        if len(values) < max(1, policy.min_samples_share * policy.samples):
            checks[name] = "UNKNOWN"
            continue
        within = sum(v <= limit for v in values) / len(values)
        checks[name] = "PASS" if within >= policy.pass_share and values[-1] <= limit else "FAIL"
    return checks


def overall(
    checks: dict[str, CheckResult], categories: list[str]
) -> Literal["RECOVERED", "NOT_RECOVERED", "INCONCLUSIVE"]:
    if "FAIL" in checks.values():
        return "NOT_RECOVERED"
    needed = [CHECK_FOR_CATEGORY[c] for c in categories if c in CHECK_FOR_CATEGORY]
    if any(checks.get(n) == "UNKNOWN" for n in needed) or "PASS" not in checks.values():
        return "INCONCLUSIVE"
    return "RECOVERED"


class Verifier:
    """Implements the shared RecoveryVerifier port (src/agents/monitoring/ports.py)."""

    def __init__(
        self,
        reader: SignalReader,
        policy: RecoveryPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._reader = reader
        self.policy = policy or RecoveryPolicy()
        self._sleep = sleep

    async def verify(self, action_id: str, context: TaskContext) -> RecoveryVerified:
        p = self.policy
        await self._sleep(p.settle_seconds)
        samples: list[dict[str, float | None]] = []
        for index in range(p.samples):
            try:
                samples.append(await self._reader.read(context.service_id))
            except Exception:  # noqa: BLE001 - a failed sample is just missing data
                samples.append({})
            if index < p.samples - 1:
                await self._sleep(p.sample_seconds)
        checks = judge(samples, p)
        raw_categories = context.signals.get("categories", [])
        categories = [str(c) for c in raw_categories] if isinstance(raw_categories, list) else []
        return RecoveryVerified(
            verification_id=new_id("VER"),
            action_id=action_id,
            window=p.window,
            status=overall(checks, categories),
            checks=dict(checks),
        )


class RecoveryWatcher:
    """Reacts to `deployment.action.completed`: verifies, then publishes the result."""

    def __init__(
        self,
        verifier: Verifier,
        publisher: EventPublisher,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._verifier = verifier
        self._publisher = publisher
        self._clock = clock

    async def on_event(self, event: EventEnvelope) -> EventEnvelope | None:
        """The recovery event sent, or None when there is nothing to verify."""
        if event.topic != "deployment.action.completed":
            return None
        action = ActionCompleted.model_validate(event.payload)
        # A failed or blocked action changed nothing that could have recovered.
        if action.status != "SUCCESS" or not action.verification_requested:
            return None
        context = TaskContext(
            service_id=event.service.service_id,
            namespace=event.service.namespace,
            environment=event.environment,
            signals={"categories": _categories(action)},
            evidence_refs=event.evidence_refs,
        )
        result = await self._verifier.verify(action.action_id, context)
        reply = build_recovery_event(
            result,
            event.service,
            Environment(event.environment),
            correlation_id=event.correlation_id,
            causation_id=event.message_id,
            severity=event.severity,
        )
        await self._publisher.publish(reply)
        return reply


def _categories(action: ActionCompleted) -> list[JsonValue]:
    """Symptom categories the action was meant to fix, when the action reports them."""
    raw = action.pre_state.get("categories")
    return [str(c) for c in raw] if isinstance(raw, list) else []
