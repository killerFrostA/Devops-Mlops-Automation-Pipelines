"""Calibrate Agent 1's thresholds on a NORMAL period of a service's real history.

The detectors are replayed on Prometheus history, minute by minute, as if the agent had been
running. On a period without incidents, every alert it would have raised is a false alert.
The script reports how high the normal signals go (memory slope, latency and CPU shifts, SLO
burn rates) and suggests thresholds that keep a safety margin above that normal noise.

Usage:
  python experiments/checks/calibrate.py fraud-api --hours 6
  python experiments/checks/calibrate.py payment-service --hours 24 --end "2026-10-20 08:00"

Choose a period WITHOUT injected faults or incidents: the result is only as good as that.
The longer the period (days, with day and night traffic), the more trustworthy the thresholds.
Output: printed table + experiments/runs/calibration/<service>.md
"""

import argparse
import asyncio
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.agents.monitoring.collectors.models import Series
from src.agents.monitoring.collectors.prometheus import (
    PrometheusClient,
    PrometheusCollector,
    build_sli_queries,
)
from src.agents.monitoring.config import load_config
from src.agents.monitoring.detectors.statistical import (
    FindingStatus,
    StatisticalPolicy,
    evaluate,
    linear_fit,
    robust_z,
)
from src.config import Settings

ROOT = Path(__file__).resolve().parents[2]
STEP_SECONDS = 60  # one replayed decision per minute


def percentile(values: list[float], share: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(share * len(ordered)) - 1)]


def replay(
    policy: StatisticalPolicy, series: dict[str, Series], start: float, end: float
) -> dict[str, Any]:
    """Run the statistical detector at every minute; collect what it measured."""
    warmup = policy.baseline_seconds + policy.recent_seconds
    alerts: dict[str, int] = {}
    slopes: list[float] = []
    shifts: dict[str, list[float]] = {}
    decisions = 0
    now = start + warmup
    while now <= end:
        decisions += 1
        for finding in evaluate(policy, series, now):
            if finding.status is FindingStatus.ANOMALY:
                alerts[finding.signal] = alerts.get(finding.signal, 0) + 1
        recent_start = now - policy.recent_seconds
        for signal, data in series.items():
            recent = [p for p in data.points if recent_start <= p[0] <= now]
            baseline = [
                v
                for t, v in data.points
                if recent_start - policy.baseline_seconds <= t < recent_start
            ]
            if signal == "memory_mb" and len(recent) >= 3:
                slopes.append(linear_fit(recent)[0])
            if len(recent) >= 3 and len(baseline) >= 3:
                current = statistics.median(v for _, v in recent)
                z = robust_z(current, baseline, policy.min_scale_fraction)
                shifts.setdefault(signal, []).append(z)
        now += STEP_SECONDS
    return {"decisions": decisions, "alerts": alerts, "slopes": slopes, "shifts": shifts}


async def burn_rates(
    collector: PrometheusCollector,
    prometheus: PrometheusClient,
    service_id: str,
    slo_objective: float,
    start: float,
    end: float,
) -> dict[str, list[float]]:
    """Error-ratio burn rate over each SLO window, sampled every minute of the period."""
    burns: dict[str, list[float]] = {}
    for window in ("1m", "5m", "1h"):
        profile = collector.profile(service_id)  # the service's own metric/label names
        query = build_sli_queries(service_id, window, "0.5", profile)["error_ratio"]
        points = await prometheus.query_range(query, start, end, STEP_SECONDS)
        burns[window] = [v / (1 - slo_objective) for _, v in points]
    return burns


async def main(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    policy = config.statistical
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d %H:%M").timestamp()
    else:
        end = datetime.now().timestamp()
    start = end - args.hours * 3600
    lookback = end - start
    async with PrometheusClient(Settings().prometheus_url) as prometheus:
        # collect_series looks back from its clock: set it to the end of the chosen period.
        collector = PrometheusCollector(
            prometheus,
            clock=lambda: datetime.fromtimestamp(end, UTC),
            profiles={s.service_id: s for s in config.services},
        )
        series = await collector.collect_series(args.service, lookback, policy.step_seconds)
        objective = config.slo.slos[0].objective
        burns = await burn_rates(collector, prometheus, args.service, objective, start, end)

    result = replay(policy, series, start, end)
    growth = policy.growth[0]
    shift_rules = {rule.signal: rule for rule in policy.level_shifts}
    slope_p99 = percentile(result["slopes"], 0.99)
    rows = [
        ("Decisions rejouees (1 par minute)", str(result["decisions"]), ""),
        (
            "Fausses alertes statistiques",
            str(sum(result["alerts"].values())),
            ", ".join(f"{k}: {v}" for k, v in result["alerts"].items()) or "aucune",
        ),
        (
            "Pente memoire normale (p99 / max)",
            f"{slope_p99 or 0:.2f} / {max(result['slopes'], default=0):.2f} Mo/min",
            f"seuil actuel {growth.min_slope_per_minute}, "
            f"conseille >= {max(growth.min_slope_per_minute, 2 * (slope_p99 or 0)):.1f}",
        ),
    ]
    for signal, values in result["shifts"].items():
        rule = shift_rules.get(signal)
        if rule is None:
            continue
        p999 = percentile(values, 0.999) or 0
        rows.append(
            (
                f"Saut normal {signal} (z robuste p99.9)",
                f"{p999:.1f}",
                f"seuil actuel {rule.min_robust_z}, "
                f"conseille >= {max(rule.min_robust_z, 1.5 * p999):.1f}",
            )
        )
    for window, values in burns.items():
        rows.append(
            (
                f"Burn rate normal sur {window} (max)",
                f"{max(values, default=0):.1f}",
                "doit rester sous le seuil de la regle qui utilise cette fenetre",
            )
        )
    period = (
        f"{datetime.fromtimestamp(start):%Y-%m-%d %H:%M} -> {datetime.fromtimestamp(end):%H:%M}"
    )
    lines = [
        f"# Calibration de l'Agent 1 : {args.service}",
        "",
        f"Periode supposee NORMALE : {period} ({args.hours} h)",
        "",
        "| Mesure | Valeur | Commentaire |",
        "|---|---|---|",
        *[f"| {a} | {b} | {c} |" for a, b, c in rows],
    ]
    out = ROOT / "experiments/runs/calibration" / f"{args.service}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n(ecrit dans {out})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("service")
    parser.add_argument("--hours", type=float, default=2)
    parser.add_argument("--end", help='end of the normal period, "YYYY-MM-DD HH:MM" (default: now)')
    parser.add_argument("--config", default=str(ROOT / "configs/monitoring.yaml"))
    asyncio.run(main(parser.parse_args()))
