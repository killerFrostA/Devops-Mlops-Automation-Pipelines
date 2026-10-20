import asyncio
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from src.agents.monitoring.collectors.models import SLIWindow
from src.agents.monitoring.collectors.prometheus import (
    PrometheusClient,
    PrometheusCollector,
    build_sli_queries,
)
from src.agents.monitoring.config import MonitoringConfig, load_config
from src.agents.monitoring.detectors.slo import (
    SLODefinition,
    SLODetector,
    SLOEvaluation,
    SLOPolicy,
    SLOStatus,
    evaluate,
)
from src.contracts.enums import Severity

WINDOWS = ("5m", "30m", "1h", "6h")


def sli(error: float | None = 0.0, slow: float | None = 0.0, rps: float | None = 5.0) -> SLIWindow:
    return SLIWindow(
        window="", request_rate_rps=rps, error_ratio=error, slow_ratio=slow, source_available=True
    )


def same_everywhere(
    error: float | None = 0.0, slow: float | None = 0.0, rps: float | None = 5.0
) -> dict[str, SLIWindow]:
    return {w: sli(error, slow, rps) for w in WINDOWS}


def find(results: list[SLOEvaluation], slo: str, rule: str) -> SLOEvaluation:
    return next(r for r in results if r.slo == slo and r.rule == rule)


def test_twenty_percent_errors_breach_the_fast_burn_rule() -> None:
    results = evaluate(SLOPolicy(), same_everywhere(error=0.20))
    fast = find(results, "availability", "fast-burn")
    assert fast.status is SLOStatus.BREACH
    assert fast.severity is Severity.HIGH
    assert fast.burn_rate_short == pytest.approx(40)  # 0.20 / (1 - 0.995)


def test_normal_traffic_is_ok() -> None:
    results = evaluate(SLOPolicy(), same_everywhere(error=0.001, slow=0.002))
    assert all(r.status is SLOStatus.OK for r in results)
    assert all(r.severity is None for r in results)


def test_finished_problem_does_not_fire() -> None:
    """High over 1h but recovered over 5m: the short window says the problem is over."""
    windows = same_everywhere(error=0.20)
    windows["5m"] = sli(error=0.0)
    assert find(evaluate(SLOPolicy(), windows), "availability", "fast-burn").status is (
        SLOStatus.OK
    )


def test_brief_spike_does_not_fire() -> None:
    """High over 5m but negligible over 1h: not yet significant for the fast-burn rule."""
    windows = same_everywhere(error=0.001)
    windows["5m"] = sli(error=0.30)
    assert find(evaluate(SLOPolicy(), windows), "availability", "fast-burn").status is (
        SLOStatus.OK
    )


def test_slow_latency_breaches_latency_slo_only() -> None:
    results = evaluate(SLOPolicy(), same_everywhere(error=0.0, slow=0.5))
    assert find(results, "latency-500ms", "fast-burn").status is SLOStatus.BREACH
    assert find(results, "availability", "fast-burn").status is SLOStatus.OK


def test_moderate_burn_triggers_only_the_slow_rule() -> None:
    results = evaluate(SLOPolicy(), same_everywhere(error=0.04))  # burn 8: >= 6, < 14.4
    assert find(results, "availability", "slow-burn").status is SLOStatus.BREACH
    assert find(results, "availability", "slow-burn").severity is Severity.MEDIUM
    assert find(results, "availability", "fast-burn").status is SLOStatus.OK


@pytest.mark.parametrize(
    "windows",
    [
        same_everywhere(error=None),  # Prometheus unavailable or no data
        same_everywhere(error=0.5, rps=0.01),  # too little traffic to trust a ratio
        same_everywhere(error=0.5, rps=None),
        {"5m": sli(error=0.5)},  # long window missing
    ],
)
def test_unknown_data_is_never_ok_or_breach(windows: dict[str, SLIWindow]) -> None:
    fast = find(evaluate(SLOPolicy(), windows), "availability", "fast-burn")
    assert fast.status is SLOStatus.INSUFFICIENT_DATA


def test_policy_rejects_invalid_objective_and_duplicates() -> None:
    with pytest.raises(ValidationError):
        SLODefinition(name="x", sli="availability", objective=1.0)
    slo = SLODefinition(name="x", sli="availability", objective=0.99)
    with pytest.raises(ValidationError, match="unique"):
        SLOPolicy(slos=(slo, slo))


def test_sli_queries_use_window_and_latency_bucket() -> None:
    queries = build_sli_queries("payment-service", "1h", "0.5")
    assert all("[1h]" in q for q in queries.values())
    assert 'le="0.5"' in queries["slow_ratio"]
    assert 'status=~"5.."' in queries["error_ratio"]


@pytest.mark.parametrize(("window", "le"), [("1h]) or vector(1", "0.5"), ("1h", '0.5"}')])
def test_sli_collection_refuses_injected_parameters(window: str, le: str) -> None:
    async def run() -> None:
        async with PrometheusClient("http://prom") as client:
            await PrometheusCollector(client).collect_sli("payment-service", [window], le)

    with pytest.raises(ValueError, match="Invalid"):
        asyncio.run(run())


def test_detector_collects_every_rule_window_then_evaluates() -> None:
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params["query"]
        asked.append(query)
        value = "5"  # request rate
        if query.startswith("(sum"):
            value = "0.2"  # error ratio
        elif query.startswith("1 -"):
            value = "0.0"  # slow ratio
        result = [{"metric": {}, "value": [0, value]}]
        return httpx.Response(
            200, json={"status": "success", "data": {"resultType": "vector", "result": result}}
        )

    async def run() -> list[SLOEvaluation]:
        http = httpx.AsyncClient(base_url="http://prom", transport=httpx.MockTransport(handler))
        async with PrometheusClient("http://prom", http=http) as client:
            return await SLODetector(PrometheusCollector(client)).evaluate("payment-service")

    results = asyncio.run(run())
    assert {w for w in WINDOWS if any(f"[{w}]" in q for q in asked)} == set(WINDOWS)
    assert find(results, "availability", "fast-burn").status is SLOStatus.BREACH
    assert find(results, "latency-500ms", "fast-burn").status is SLOStatus.OK


def project_config() -> MonitoringConfig:
    return load_config(Path(__file__).resolve().parents[2] / "configs/monitoring.yaml")


def ratios(**by_window: float) -> dict[str, SLIWindow]:
    """Error ratio per window, e.g. ratios(m1=1.0, m5=0.2) for the 1m and 5m windows."""
    return {key[1:] + key[0]: sli(error=value) for key, value in by_window.items()}


def test_project_very_fast_rule_catches_a_violent_fault_in_about_a_minute() -> None:
    """One minute of total outage: 20 % of the last 5 min, the 1h rule is still far below."""
    data = ratios(m1=1.0, m5=0.2, m30=1 / 30, h1=1 / 60, h6=1 / 360)
    results = evaluate(project_config().slo, data)
    very_fast = find(results, "availability", "very-fast")
    assert (very_fast.status, very_fast.severity) == (SLOStatus.BREACH, Severity.HIGH)
    assert find(results, "availability", "fast-burn").status is SLOStatus.OK


def test_project_very_fast_rule_ignores_a_short_blip() -> None:
    """20 s of errors: 6.7 % of the last 5 min is a burn of 13, below 36."""
    data = ratios(m1=1 / 3, m5=1 / 15, m30=1 / 90, h1=1 / 180, h6=1 / 1080)
    results = evaluate(project_config().slo, data)
    assert find(results, "availability", "very-fast").status is SLOStatus.OK


def test_project_config_detects_sooner() -> None:
    config = project_config()
    assert config.interval_seconds == 30
    assert config.availability.down_after_seconds == 45
    assert "very-fast" in {rule.name for rule in config.slo.rules}
