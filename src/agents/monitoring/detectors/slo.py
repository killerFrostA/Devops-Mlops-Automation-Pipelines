"""Layer 1: SLO detection with multi-window, multi-burn-rate rules (Google SRE Workbook, ch. 5).

An SLO promises a share of good events, e.g. 99.5 %. The error budget is the rest (0.5 %).
The burn rate is how fast that budget is being spent: bad_ratio / (1 - objective).
Burn rate 1 spends exactly the budget over the SLO period; 14.4 spends 2 % of a 30-day budget
in one hour. A rule fires only when BOTH its long window (the problem is significant) and its
short window (the problem is still happening) exceed the threshold.

The detector reports observed facts (which SLO, how fast, over which windows), never a cause.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.agents.monitoring.collectors.models import SLIWindow
from src.contracts.enums import Severity


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SLODefinition(_Config):
    name: str
    sli: Literal["availability", "latency"]
    objective: float = Field(gt=0, lt=1)  # e.g. 0.995 = 99.5 % good events


class BurnRateRule(_Config):
    name: str
    long_window: str
    short_window: str
    burn_rate: float = Field(gt=0)
    severity: Severity


class SLOPolicy(_Config):
    """Thresholds are configuration, not code. Defaults follow the SRE Workbook and must be
    validated experimentally on the test bench before being trusted."""

    slos: tuple[SLODefinition, ...] = (
        SLODefinition(name="availability", sli="availability", objective=0.995),
        SLODefinition(name="latency-500ms", sli="latency", objective=0.995),
    )
    rules: tuple[BurnRateRule, ...] = (
        BurnRateRule(
            name="fast-burn",
            long_window="1h",
            short_window="5m",
            burn_rate=14.4,
            severity=Severity.HIGH,
        ),
        BurnRateRule(
            name="slow-burn",
            long_window="6h",
            short_window="30m",
            burn_rate=6.0,
            severity=Severity.MEDIUM,
        ),
    )
    # Must equal a histogram bucket bound of the service ("0.5" = 500 ms).
    latency_le: str = "0.5"
    # Below this traffic a few failures swing the ratio wildly: report INSUFFICIENT_DATA.
    min_request_rate_rps: float = Field(default=0.05, ge=0)

    @model_validator(mode="after")
    def unique_names(self) -> "SLOPolicy":
        if len({s.name for s in self.slos}) != len(self.slos):
            raise ValueError("SLO names must be unique")
        if len({r.name for r in self.rules}) != len(self.rules):
            raise ValueError("Rule names must be unique")
        return self

    @property
    def windows(self) -> set[str]:
        return {w for r in self.rules for w in (r.long_window, r.short_window)}


class SLOStatus(StrEnum):
    BREACH = "BREACH"
    OK = "OK"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True, slots=True)
class SLOEvaluation:
    slo: str
    rule: str
    status: SLOStatus
    burn_rate_long: float | None
    burn_rate_short: float | None
    threshold: float
    severity: Severity | None  # set only on BREACH


def _bad_ratio(window: SLIWindow, sli: str) -> float | None:
    return window.error_ratio if sli == "availability" else window.slow_ratio


def evaluate(policy: SLOPolicy, windows: dict[str, SLIWindow]) -> list[SLOEvaluation]:
    """Pure decision: one evaluation per (SLO, rule). No I/O, so it is fully unit-testable."""
    results: list[SLOEvaluation] = []
    for slo in policy.slos:
        budget = 1 - slo.objective
        for rule in policy.rules:
            long, short = windows.get(rule.long_window), windows.get(rule.short_window)
            ratios = [_bad_ratio(w, slo.sli) if w else None for w in (long, short)]
            burns = [None if r is None else r / budget for r in ratios]
            enough_traffic = (
                short is not None
                and short.request_rate_rps is not None
                and short.request_rate_rps >= policy.min_request_rate_rps
            )
            if not enough_traffic or None in burns:
                status = SLOStatus.INSUFFICIENT_DATA
            elif all(b is not None and b >= rule.burn_rate for b in burns):
                status = SLOStatus.BREACH
            else:
                status = SLOStatus.OK
            results.append(
                SLOEvaluation(
                    slo=slo.name,
                    rule=rule.name,
                    status=status,
                    burn_rate_long=burns[0],
                    burn_rate_short=burns[1],
                    threshold=rule.burn_rate,
                    severity=rule.severity if status is SLOStatus.BREACH else None,
                )
            )
    return results


class SLISource(Protocol):
    async def collect_sli(
        self, service_id: str, windows: set[str], latency_le: str
    ) -> dict[str, SLIWindow]: ...


class SLODetector:
    def __init__(self, source: SLISource, policy: SLOPolicy | None = None) -> None:
        self._source = source
        self.policy = policy or SLOPolicy()

    async def evaluate(self, service_id: str) -> list[SLOEvaluation]:
        windows = await self._source.collect_sli(
            service_id, self.policy.windows, self.policy.latency_le
        )
        return evaluate(self.policy, windows)
