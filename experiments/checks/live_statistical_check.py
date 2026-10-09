"""Live check of the statistical detector (step 4) against the running test bench.

Usage:
  python experiments/checks/live_statistical_check.py            # one evaluation
  python experiments/checks/live_statistical_check.py --watch 300  # every 15 s for 300 s,
                                                                   # printing detection delays
The bench policy uses a 10 min baseline instead of 30 min: the bench has little history.
"""

import argparse
import asyncio
import time

from src.agents.monitoring.collectors.prometheus import PrometheusClient, PrometheusCollector
from src.agents.monitoring.detectors.statistical import (
    FindingStatus,
    StatisticalDetector,
    StatisticalFinding,
    StatisticalPolicy,
)

PROMETHEUS_URL = "http://127.0.0.1:9090"
SERVICE = "payment-service"
BENCH_POLICY = StatisticalPolicy(baseline_seconds=600)


def describe(f: StatisticalFinding) -> str:
    def num(value: float | None, unit: str = "") -> str:
        return "-" if value is None else f"{value:.2f}{unit}"

    if f.method == "growth":
        detail = f"pente={num(f.slope_per_minute, '/min')} r2={num(f.r2)}"
    else:
        change = None if f.relative_change is None else f.relative_change * 100
        detail = (
            f"actuel={num(f.current)} habituel={num(f.baseline_median)} "
            f"z={num(f.robust_z)} variation={num(change, '%')}"
        )
    severity = f" ({f.severity})" if f.severity else ""
    return f"   {f.signal:15} {f.method:11} {detail:62} -> {f.status}{severity}"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", type=float, default=0, help="seconds to keep evaluating")
    args = parser.parse_args()

    async with PrometheusClient(PROMETHEUS_URL, timeout_seconds=3) as client:
        detector = StatisticalDetector(PrometheusCollector(client), BENCH_POLICY)
        started = time.time()
        first_seen: dict[str, float] = {}
        while True:
            findings = await detector.evaluate(SERVICE)
            elapsed = time.time() - started
            print(f"== Etape 4 - detection statistique ({SERVICE}) t+{elapsed:.0f}s")
            for finding in findings:
                print(describe(finding))
                if finding.status is FindingStatus.ANOMALY:
                    first_seen.setdefault(finding.signal, elapsed)
            if elapsed >= args.watch:
                break
            await asyncio.sleep(15)
        for signal, seconds in first_seen.items():
            print(f"   premiere detection {signal}: t+{seconds:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
