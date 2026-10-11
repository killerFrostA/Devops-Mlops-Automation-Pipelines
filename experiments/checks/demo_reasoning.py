"""Simulated demo of the reasoning layer (no Docker needed).

Scenario: payment-api latency rises, a new "connection pool exhausted" log appears on
payment-api, and payments-db memory grows. The agent runs its real loop on these simulated
detections and should rank payments-db as the most probable origin.

Usage: python experiments/checks/demo_reasoning.py
"""

import asyncio
from pathlib import Path

from src.agents.monitoring.config import MonitoringConfig, WatchedService
from src.agents.monitoring.decision import DecisionMaker, DecisionPolicy
from src.agents.monitoring.detectors.logs import FindingKind, LogFinding, LogLabel
from src.agents.monitoring.detectors.slo import SLOEvaluation
from src.agents.monitoring.detectors.statistical import FindingStatus, StatisticalFinding
from src.agents.monitoring.grouping import InMemoryIncidentStore, SymptomGrouper
from src.agents.monitoring.loop import ConsoleReporter, MonitoringLoop
from src.agents.monitoring.reasoning import Reasoner, load_reasoning_config
from src.contracts.enums import Severity

ROOT = Path(__file__).resolve().parents[2]

STATISTICAL = {
    "payment-api": [
        StatisticalFinding(
            "latency_p95_ms",
            "latency",
            "level_shift",
            FindingStatus.ANOMALY,
            current=640.0,
            baseline_median=95.0,
            robust_z=42.0,
            relative_change=5.7,
            severity=Severity.MEDIUM,
        )
    ],
    "payments-db": [
        StatisticalFinding(
            "memory_mb",
            "memory_pressure",
            "growth",
            FindingStatus.ANOMALY,
            slope_per_minute=35.0,
            r2=0.98,
            severity=Severity.MEDIUM,
        )
    ],
}
LOGS = {
    "payment-api": [
        LogFinding(
            FindingKind.NEW_PATTERN,
            "ERROR HikariPool-1 connection pool exhausted, waited <*> ms",
            LogLabel.UNKNOWN,
            "none",
            85,
            0,
            "ERROR HikariPool-1 connection pool exhausted, waited 30000 ms",
            True,
        )
    ]
}


class SimulatedSignals:
    """Extra metrics the decision layer reads when it hesitates."""

    async def read(self, service_id: str) -> dict[str, float | None]:
        if service_id == "payments-db":
            return {"memory_slope_mb_min": 35.0, "error_rate": 0.0}
        return {"latency_p95_ms": 640.0, "error_rate": 0.0}


class Scripted[T]:
    def __init__(self, results: dict[str, list[T]]) -> None:
        self.results = results

    async def evaluate(self, service_id: str) -> list[T]:
        return self.results.get(service_id, [])


async def main() -> None:
    config = MonitoringConfig(
        services=(
            WatchedService(service_id="payment-api", namespace="prod"),
            WatchedService(service_id="payments-db", namespace="prod"),
        )
    )
    no_slo: Scripted[SLOEvaluation] = Scripted({})
    loop = MonitoringLoop(
        config=config,
        slo=no_slo,
        statistical=Scripted(STATISTICAL),
        logs=Scripted(LOGS),
        grouper=SymptomGrouper(InMemoryIncidentStore()),
        reporter=ConsoleReporter(),
        reasoner=Reasoner(load_reasoning_config(ROOT / "configs/dependencies.yaml")),
        decider=DecisionMaker(DecisionPolicy(), SimulatedSignals()),
    )
    print("Scenario simule : latence API + log 'connection pool exhausted' + RAM base anormale")
    await loop.run_cycle()


if __name__ == "__main__":
    asyncio.run(main())
