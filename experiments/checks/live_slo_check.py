"""Live check of Agent 1 against the running test bench (Prometheus on 127.0.0.1:9090).

Prints the collector snapshot (step 2) and the SLO evaluation (step 3) for payment-service.
Usage: python experiments/checks/live_slo_check.py
"""

import asyncio

from src.agents.monitoring.collectors.prometheus import PrometheusClient, PrometheusCollector
from src.agents.monitoring.detectors.slo import BurnRateRule, SLODetector, SLOPolicy
from src.contracts.enums import Severity

PROMETHEUS_URL = "http://127.0.0.1:9090"
SERVICE = "payment-service"

# Same threshold as production, shorter windows: the bench has minutes of history, not hours.
BENCH_POLICY = SLOPolicy(
    rules=(
        BurnRateRule(
            name="bench-fast",
            long_window="10m",
            short_window="1m",
            burn_rate=14.4,
            severity=Severity.HIGH,
        ),
    )
)


def fmt(burn: float | None) -> str:
    return "   - " if burn is None else f"{burn:5.1f}"


async def main() -> None:
    async with PrometheusClient(PROMETHEUS_URL, timeout_seconds=3) as client:
        collector = PrometheusCollector(client)

        print(f"== Etape 2 - collecteur ({SERVICE})")
        snapshot = await collector.collect(SERVICE)
        for name in ("memory_mb", "memory_slope_mb_min", "request_rate_rps", "error_rate"):
            print(f"   {name:22} {getattr(snapshot, name)}")
        print(f"   {'latency_p95_ms':22} {snapshot.latency_p95_ms}")
        print(f"   {'source_available':22} {snapshot.source_available}")

        for label, policy in (
            ("defaut (1h/5m, 6h/30m)", SLOPolicy()),
            ("banc (10m/1m)", BENCH_POLICY),
        ):
            print(f"== Etape 3 - SLO, politique {label}")
            for r in await SLODetector(collector, policy).evaluate(SERVICE):
                severity = f" ({r.severity})" if r.severity else ""
                print(
                    f"   {r.slo:15} {r.rule:11} burn long={fmt(r.burn_rate_long)} "
                    f"court={fmt(r.burn_rate_short)} seuil={r.threshold:4.1f} "
                    f"-> {r.status}{severity}"
                )


if __name__ == "__main__":
    asyncio.run(main())
