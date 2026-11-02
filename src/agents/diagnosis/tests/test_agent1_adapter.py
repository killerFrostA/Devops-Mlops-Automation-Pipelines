from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from datetime import UTC, datetime, timedelta

import pytest

from src.agents.diagnosis.contracts import DiagnosisAgentResult, DiagnosisInput
from src.agents.diagnosis.service import DiagnosisAgent
from src.agents.diagnosis import rules
from src.agents.diagnosis.signal_adapter import adapt_agent1_signals
from src.contracts.enums import ActionType, AgentId, Environment, ResultStatus
from src.contracts.tasks import AgentTask, TaskContext


FIXTURES = Path(__file__).parent / "fixtures"


def _diagnose_fixture(
    signals: dict[str, object],
    *,
    recent_change_events: list[dict[str, object]] | None = None,
) -> tuple[DiagnosisAgentResult, DiagnosisInput]:
    payload = DiagnosisInput(
        incident_id="INC-2026-25bc2c7bae",
        service_id="payment-service",
        namespace="local",
        environment=Environment.LOCAL,
        issue_summary="Agent 1 anomaly event for payment-service.",
        signals=signals,
        recent_change_events=recent_change_events or [],
    )
    task = AgentTask(
        task_id="TASK-AGENT1-ADAPTER",
        incident_id=payload.incident_id,
        agent_id=AgentId.DIAGNOSIS,
        objective=payload.issue_summary,
        task_context=TaskContext(
            service_id=payload.service_id,
            namespace=payload.namespace,
            environment=payload.environment,
        ),
        deadline=datetime.now(UTC) + timedelta(minutes=5),
    )
    result = DiagnosisAgent().diagnose_input(task, payload)
    assert isinstance(result, DiagnosisAgentResult)
    assert result.status is ResultStatus.SUCCEEDED
    return result, payload


@pytest.mark.parametrize(
    ("filename", "expected_cause"),
    [
        ("agent1_memory_oom.json", "MEMORY_LIMIT_REACHED"),
        ("agent1_memory_pressure.json", "MEMORY_GROWTH"),
    ],
)
def test_agent1_fixtures_adapt_end_to_end_without_mutation(
    filename: str,
    expected_cause: str,
) -> None:
    fixture_path = FIXTURES / filename
    original_bytes = fixture_path.read_bytes()
    message = json.loads(original_bytes)
    signals = message["payload"]["signals"]
    original_signals = deepcopy(signals)

    result, payload = _diagnose_fixture(signals)
    adaptation = adapt_agent1_signals(payload)

    assert result.diagnosis.primary_root_cause == expected_cause
    # MEMORY_GROWTH is symptom-level; reaching the memory limit is itself actionable.
    assert result.diagnosis.requires_additional_evidence is (expected_cause == "MEMORY_GROWTH")
    assert all(
        action.verification_profile == "payment-service-slo-default"
        for action in result.diagnosis.recommended_actions
    )
    assert not payload.recent_change_events
    assert not adaptation.payload.recent_change_events
    assert signals == original_signals
    assert fixture_path.read_bytes() == original_bytes
    assert any("first_seen" in item and "detection time" in item for item in result.diagnosis.limitations)

    if filename == "agent1_memory_oom.json":
        assert adaptation.payload.signals["memory_percent"] == pytest.approx(99.98)
        assert adaptation.payload.signals["restart_count"] == 1
        assert adaptation.payload.memory_rise_started_at is not None
        assert any(
            item.value.get("memory_percent") == pytest.approx(99.98)
            for item in result.evidence
        )
        ratio_ref = next(
            item.evidence_id
            for item in result.evidence
            if item.value.get("agent1_fact_key") == "events.memory_peak_ratio"
        )
        restart_ref = next(
            item.evidence_id
            for item in result.evidence
            if item.value.get("agent1_fact_key") == "events.restarts"
        )
        assert {ratio_ref, restart_ref} <= set(result.diagnosis.hypotheses[0].evidence_refs)
    else:
        assert adaptation.payload.memory_growth_observed is True
        assert adaptation.payload.signals["statistical.slope_per_minute"] == pytest.approx(126.9)
        assert adaptation.payload.signals["statistical.slope_unit"] == "Mo/min"
        assert "memory_percent" not in adaptation.payload.signals
        assert any("memory in Mo" in item for item in result.diagnosis.limitations)
        assert adaptation.payload.memory_rise_started_at is not None
        fact_keys = {
            item.value.get("agent1_fact_key")
            for item in result.evidence
            if item.evidence_id in result.diagnosis.hypotheses[0].evidence_refs
        }
        assert {
            "statistical.current",
            "statistical.baseline_median",
            "statistical.slope_per_minute",
        } <= fact_keys
        actions = [action.action for action in result.diagnosis.recommended_actions]
        assert ActionType.ROLLBACK not in actions
        assert actions == [ActionType.ROLLING_RESTART]
        assert any("deployment history" in item.lower() for item in result.diagnosis.limitations)


def test_agent1_memory_pressure_with_deployment_ten_minutes_before_detection_is_leak() -> None:
    signals = json.loads((FIXTURES / "agent1_memory_pressure.json").read_text(encoding="utf-8"))[
        "payload"
    ]["signals"]
    incident_first_seen = datetime.fromisoformat(
        signals["incidents"][0]["first_seen"].replace("Z", "+00:00")
    )
    deployment_at = incident_first_seen - timedelta(minutes=10)

    result, _ = _diagnose_fixture(
        signals,
        recent_change_events=[
            {
                "change_id": "DEPLOY-773",
                "description": "Deployment DEP-773",
                "changed_at": deployment_at,
            }
        ],
    )

    assert result.diagnosis.primary_root_cause == "MEMORY_LEAK_POST_DEPLOY"
    # A dated deployment before the rise supports acting (architecture example 12.1).
    assert result.diagnosis.requires_additional_evidence is False
    assert any(
        item.evidence_type == "git_change"
        and item.evidence_id in result.diagnosis.hypotheses[0].evidence_refs
        for item in result.evidence
    )
    actions = result.diagnosis.recommended_actions
    assert [action.action for action in actions] == [ActionType.ROLLBACK, ActionType.SCALE_REPLICAS]
    assert actions[0].parameters["deployment_id"] == "DEPLOY-773"


def test_agent1_memory_pressure_deployment_two_minutes_before_detection_is_uncertain() -> None:
    signals = json.loads((FIXTURES / "agent1_memory_pressure.json").read_text(encoding="utf-8"))[
        "payload"
    ]["signals"]
    incident_first_seen = datetime.fromisoformat(
        signals["incidents"][0]["first_seen"].replace("Z", "+00:00")
    )
    deployment_at = incident_first_seen - timedelta(minutes=2)

    result, _ = _diagnose_fixture(
        signals,
        recent_change_events=[
            {
                "change_id": "DEPLOY-773",
                "description": "Deployment DEP-773",
                "changed_at": deployment_at,
            }
        ],
    )

    assert result.diagnosis.primary_root_cause == "MEMORY_GROWTH"
    assert result.diagnosis.requires_additional_evidence is True
    assert any("less than the 5-minute detection margin" in item for item in result.explainability.missing_information)
    assert any("uncertainty margin" in item for item in result.diagnosis.limitations)
    assert ActionType.ROLLBACK not in [
        action.action for action in result.diagnosis.recommended_actions
    ]


def test_agent1_slope_threshold_can_be_mutated(monkeypatch: pytest.MonkeyPatch) -> None:
    signals = json.loads((FIXTURES / "agent1_memory_pressure.json").read_text(encoding="utf-8"))[
        "payload"
    ]["signals"]

    monkeypatch.setattr(rules, "AGENT1_MEMORY_GROWTH_MIN_SLOPE_MO_PER_MIN", 126.9)
    at_threshold, _ = _diagnose_fixture(signals)
    assert at_threshold.diagnosis.primary_root_cause == "MEMORY_GROWTH"

    monkeypatch.setattr(rules, "AGENT1_MEMORY_GROWTH_MIN_SLOPE_MO_PER_MIN", 126.91)
    above_threshold, _ = _diagnose_fixture(signals)
    assert above_threshold.diagnosis.primary_root_cause == "INSUFFICIENT_EVIDENCE"


def test_agent1_oom_ratio_threshold_can_be_mutated(monkeypatch: pytest.MonkeyPatch) -> None:
    signals = json.loads((FIXTURES / "agent1_memory_oom.json").read_text(encoding="utf-8"))[
        "payload"
    ]["signals"]

    monkeypatch.setattr(rules, "AGENT1_MEMORY_OOM_MIN_PEAK_RATIO", 0.9998)
    at_threshold, _ = _diagnose_fixture(signals)
    assert at_threshold.diagnosis.primary_root_cause == "MEMORY_LIMIT_REACHED"

    monkeypatch.setattr(rules, "AGENT1_MEMORY_OOM_MIN_PEAK_RATIO", 0.99981)
    above_threshold, _ = _diagnose_fixture(signals)
    assert above_threshold.diagnosis.primary_root_cause != "MEMORY_LIMIT_REACHED"


def test_agent1_conversions_and_pdf_aliases() -> None:
    payload = DiagnosisInput(
        incident_id="INC-ALIAS",
        service_id="payment-service",
        namespace="local",
        environment=Environment.LOCAL,
        issue_summary="Alias and unit conversion test.",
        signals={
            "categories": ["memory_oom"],
            "memory_pct": 94,
            "latency_p95_ms": 1200,
            "pod_restarts": 3,
            "incidents": [
                {
                    "category": "memory_oom",
                    "facts": {
                        "events.memory_peak_ratio": 0.94,
                        "events.restarts": 3,
                        "slo.burn_rate_long": 2,
                        "statistical.signal": "latency_p95_ms",
                        "statistical.current": 1200,
                    },
                }
            ],
        },
    )

    adapted = adapt_agent1_signals(payload)

    assert adapted.payload.signals["memory_percent"] == pytest.approx(94)
    assert adapted.payload.signals["latency_ms"] == pytest.approx(1200)
    assert adapted.payload.signals["restart_count"] == pytest.approx(3)
    assert adapted.payload.signals["error_rate"] == pytest.approx(0.01)
    assert any("explicit factor 0.005" in item for item in adapted.limitations)


def test_agent1_conflicts_unknown_values_and_ambiguous_units_are_reported() -> None:
    payload = DiagnosisInput(
        incident_id="INC-LIMITATIONS",
        service_id="payment-service",
        namespace="local",
        environment=Environment.LOCAL,
        issue_summary="Conflict and unknown signal test.",
        signals={
            "categories": ["memory_oom", "new_category"],
            "memory_pct": 94,
            "incidents": [
                {
                    "category": "memory_oom",
                    "facts": {
                        "events.memory_peak_ratio": 0.8,
                        "events.unrecognized_metric": 12,
                    },
                }
            ],
            "suspects": [{"service_id": "payment-service", "probability": 0.9, "new_field": 1}],
            "unrecognized_signal": 42,
        },
    )

    adapted = adapt_agent1_signals(payload)

    assert "memory_percent" not in adapted.payload.signals
    assert any("Conflicting sources for 'memory_percent'" in item for item in adapted.limitations)
    assert any("new_category" in item for item in adapted.limitations)
    assert any("events.unrecognized_metric" in item for item in adapted.limitations)
    assert any("unrecognized_signal" in item for item in adapted.limitations)
    assert any("suspects[0].new_field" in item for item in adapted.limitations)
    assert payload.signals["memory_pct"] == 94


def test_memory_pct_fraction_is_not_interpreted_as_a_percentage() -> None:
    payload = DiagnosisInput(
        incident_id="INC-AMBIGUOUS-UNIT",
        service_id="payment-service",
        namespace="local",
        environment=Environment.LOCAL,
        issue_summary="Ambiguous memory unit test.",
        signals={"memory_pct": 0.94},
    )

    adapted = adapt_agent1_signals(payload)

    assert "memory_percent" not in adapted.payload.signals
    assert any("could be a fraction or a percentage" in item for item in adapted.limitations)


def test_cpu_cores_remain_raw_context_without_invented_percentage() -> None:
    payload = DiagnosisInput(
        incident_id="INC-CPU-CORES",
        service_id="payment-service",
        namespace="local",
        environment=Environment.LOCAL,
        issue_summary="CPU cores without a resource limit.",
        signals={
            "categories": ["cpu_pressure"],
            "incidents": [
                {
                    "category": "cpu_pressure",
                    "facts": {
                        "statistical.signal": "cpu_cores",
                        "statistical.current": 2.5,
                    },
                }
            ],
        },
    )

    result, _ = _diagnose_fixture(payload.signals)
    adapted = adapt_agent1_signals(payload)

    assert any("CPU in cores" in item for item in result.diagnosis.limitations)
    assert "cpu_percent" not in adapted.payload.signals
    assert adapted.payload.signals["agent1_context"]["incidents"][0]["facts"][
        "statistical.current"
    ] == 2.5
    assert any("CPU in cores" in item and "no CPU limit" in item for item in adapted.limitations)
