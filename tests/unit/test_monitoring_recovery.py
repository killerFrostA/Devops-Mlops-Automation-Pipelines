import asyncio
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]

from src.agents.monitoring.recovery import (
    RecoveryPolicy,
    RecoveryWatcher,
    Verifier,
    judge,
    overall,
)
from src.contracts.enums import Environment
from src.contracts.events import EventEnvelope
from src.contracts.payloads import RecoveryVerified
from src.contracts.tasks import TaskContext

ROOT = Path(__file__).resolve().parents[2]
POLICY = RecoveryPolicy(settle_seconds=30, window_seconds=120, sample_seconds=30)  # 4 samples
HEALTHY: dict[str, float | None] = {
    "error_rate": 0.0,
    "latency_p95_ms": 90.0,
    "memory_slope_mb_min": 0.1,
}
LEAKING: dict[str, float | None] = {
    "error_rate": 0.0,
    "latency_p95_ms": 90.0,
    "memory_slope_mb_min": 30.0,
}


class Script:
    """Returns the scripted samples in order; records the waits instead of sleeping."""

    def __init__(self, samples: list[dict[str, float | None]]) -> None:
        self.samples = samples
        self.waits: list[float] = []

    async def read(self, service_id: str) -> dict[str, float | None]:
        sample = self.samples.pop(0)
        if not sample:
            raise ConnectionError("prometheus down")
        return sample

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)


def verify(
    samples: list[dict[str, float | None]], categories: list[str] | None = None
) -> tuple[RecoveryVerified, Script]:
    script = Script(samples)
    context = TaskContext(
        service_id="payment-service",
        namespace="local",
        environment=Environment.LOCAL,
        signals={"categories": list(categories or [])},
    )
    verifier = Verifier(script, POLICY, sleep=script.sleep)
    return asyncio.run(verifier.verify("ACT-1", context)), script


def test_healthy_after_the_action_is_recovered() -> None:
    result, script = verify([HEALTHY] * 4, ["memory_pressure"])
    assert result.status == "RECOVERED"
    assert result.checks == {
        "error_rate": "PASS",
        "latency_p95_ms": "PASS",
        "memory_slope_mb_min": "PASS",
    }
    assert result.window == "2m"
    assert script.waits == [30, 30, 30, 30]  # settle, then between the 4 samples


def test_leak_still_there_is_not_recovered() -> None:
    result, _ = verify([HEALTHY, LEAKING, LEAKING, LEAKING], ["memory_pressure"])
    assert result.status == "NOT_RECOVERED"
    assert result.checks["memory_slope_mb_min"] == "FAIL"


def test_problem_back_at_the_end_fails_even_if_mostly_fine() -> None:
    result, _ = verify([HEALTHY, HEALTHY, HEALTHY, HEALTHY][:3] + [LEAKING])
    assert result.checks["memory_slope_mb_min"] == "FAIL"  # 75 % fine, but the last is bad


def test_no_traffic_means_the_reported_error_problem_cannot_be_confirmed_gone() -> None:
    quiet = {"error_rate": None, "latency_p95_ms": None, "memory_slope_mb_min": 0.1}
    result, _ = verify([quiet] * 4, ["error_rate"])
    assert result.checks["error_rate"] == "UNKNOWN"
    assert result.status == "INCONCLUSIVE"


def test_unknown_unrelated_check_does_not_block_a_recovery() -> None:
    container = {"error_rate": None, "latency_p95_ms": None, "memory_slope_mb_min": 0.0}
    result, _ = verify([container] * 4, ["memory_pressure"])  # container mode: no requests
    assert result.status == "RECOVERED"


def test_source_down_during_the_window_is_inconclusive_not_a_crash() -> None:
    result, _ = verify([{}, {}, {}, {}], ["memory_pressure"])
    assert set(result.checks.values()) == {"UNKNOWN"}
    assert result.status == "INCONCLUSIVE"


def test_judge_and_overall_rules() -> None:
    assert judge([HEALTHY], RecoveryPolicy(window_seconds=30, sample_seconds=30))["error_rate"] == (
        "PASS"
    )
    assert overall({"a": "PASS", "b": "FAIL"}, []) == "NOT_RECOVERED"
    assert overall({"a": "UNKNOWN"}, []) == "INCONCLUSIVE"


def action_event(status: str = "SUCCESS", verification: bool = True) -> EventEnvelope:
    path = ROOT / "contracts/examples/v1/deployment.action.completed.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"].update(
        status=status,
        verification_requested=verification,
        pre_state={"categories": ["memory_pressure"]},
    )
    return EventEnvelope.model_validate(document)


class Collecting:
    def __init__(self) -> None:
        self.events: list[EventEnvelope] = []

    async def publish(self, event: EventEnvelope) -> None:
        self.events.append(event)


def test_watcher_verifies_a_successful_action_and_answers_on_the_contract() -> None:
    script = Script([HEALTHY] * 4)
    sink = Collecting()
    watcher = RecoveryWatcher(Verifier(script, POLICY, sleep=script.sleep), sink)
    reply = asyncio.run(watcher.on_event(action_event()))
    assert reply is not None and sink.events == [reply]
    schema_path = ROOT / "contracts/jsonschema/v1/EventEnvelope.schema.json"
    Draft202012Validator(
        json.loads(schema_path.read_text()), format_checker=FormatChecker()
    ).validate(reply.model_dump(mode="json"))
    original = action_event()
    assert reply.topic == "monitoring.recovery.verified"
    assert reply.correlation_id == original.correlation_id  # same incident
    assert reply.causation_id == original.message_id  # answer to that action
    assert RecoveryVerified.model_validate(reply.payload).status == "RECOVERED"


def test_watcher_ignores_failed_actions_and_other_topics() -> None:
    sink = Collecting()
    script = Script([])
    watcher = RecoveryWatcher(Verifier(script, POLICY, sleep=script.sleep), sink)
    assert asyncio.run(watcher.on_event(action_event(status="FAILED"))) is None
    assert asyncio.run(watcher.on_event(action_event(verification=False))) is None
    anomaly = json.loads(
        (ROOT / "contracts/examples/v1/monitoring.anomaly.detected.json").read_text()
    )
    assert asyncio.run(watcher.on_event(EventEnvelope.model_validate(anomaly))) is None
    assert sink.events == []
