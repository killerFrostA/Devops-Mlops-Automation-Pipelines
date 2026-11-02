from datetime import UTC, datetime

from src.agents.diagnosis.contracts import DiagnosisInput
from src.agents.diagnosis.evidence_collector import collect_evidence


def test_collect_evidence_builds_traceable_metric_and_release_items() -> None:
    payload = DiagnosisInput(
        incident_id="INC-001",
        service_id="payment-api",
        namespace="prod",
        environment="production",
        issue_summary="Memory usage is rising and pods restart after deployment.",
        signals={"memory_percent": 84.0, "restart_count": 5, "latency_ms": 980},
        recent_changes=["deploy v2.6", "config change on limits"],
        evidence=[],
        constraints=["Do not execute rollback automatically."],
    )

    collected = collect_evidence(payload)

    assert len(collected) >= 3
    assert any(item.evidence_type == "metric" for item in collected)
    assert any(item.evidence_type == "git_change" for item in collected)
    assert all(item.evidence_id for item in collected)


def test_collect_evidence_returns_empty_list_for_clean_payload() -> None:
    payload = DiagnosisInput(
        incident_id="INC-002",
        service_id="auth-api",
        namespace="prod",
        environment="production",
        issue_summary="No anomalies observed.",
        signals={"memory_percent": 21.0, "cpu_percent": 12.0},
        recent_changes=[],
        evidence=[],
        constraints=[],
    )

    assert collect_evidence(payload) == []


def test_derived_metrics_are_not_mislabeled_as_live_source_data() -> None:
    payload = DiagnosisInput(
        incident_id="INC-DERIVED",
        service_id="payment-api",
        namespace="test",
        environment="test",
        issue_summary="A supplied metric indicates high memory usage.",
        signals={"memory_percent": 84, "restart_count": 3},
    )

    evidence = collect_evidence(payload)

    assert evidence
    assert all(item.source == "diagnosis_input" for item in evidence)
    assert all(item.provenance == "derived" for item in evidence)
    assert all(
        item.observed_at is None and item.confidence is None
        for item in evidence
        if item.evidence_type != "metric" or "memory_growth_observed" not in item.value
    )
    assert all(item.evidence_type != "k8s_event" for item in evidence)


def test_recorded_memory_growth_preserves_its_supplied_timestamp() -> None:
    rise_started_at = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
    payload = DiagnosisInput(
        incident_id="INC-GROWTH-EVIDENCE",
        service_id="payment-api",
        namespace="test",
        environment="test",
        issue_summary="Memory growth was recorded.",
        memory_rise_started_at=rise_started_at,
        memory_growth_observed=True,
    )

    growth_evidence = [
        item
        for item in collect_evidence(payload)
        if item.evidence_id == "EVD-INC-GROWTH-EVIDENCE-MEMORY-GROWTH"
    ]

    assert len(growth_evidence) == 1
    assert growth_evidence[0].observed_at == rise_started_at
    assert growth_evidence[0].value == {"memory_growth_observed": True}


def test_fixture_operational_records_remain_explicitly_synthetic(
    operational_evidence_fixture,
) -> None:
    payload = DiagnosisInput(
        incident_id="INC-FIXTURE",
        service_id="payment-api",
        namespace="test",
        environment="test",
        issue_summary="A fixture-backed memory incident.",
        signals={"memory_percent": 88, "restart_count": 3},
        evidence=operational_evidence_fixture,
    )

    evidence = collect_evidence(payload)

    assert {item.evidence_type for item in evidence} >= {
        "metric",
        "log",
        "k8s_event",
        "git_change",
    }
    fixtures = [item for item in evidence if item.provenance == "fixture"]
    assert len(fixtures) == 4
    assert all(item.source.startswith("fixture://") for item in fixtures)
