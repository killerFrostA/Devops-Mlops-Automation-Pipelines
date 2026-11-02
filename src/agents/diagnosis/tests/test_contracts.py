from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.agents.diagnosis.contracts import DiagnosisInput, EvidenceItem


def test_diagnosis_input_accepts_incident_signals_and_evidence() -> None:
    evidence = EvidenceItem(
        evidence_id="EVD-METRIC-01",
        evidence_type="metric",
        source="prometheus",
        observed_at=datetime.now(UTC),
        summary="memory_usage increased by 40%",
        value={"memory_percent": 84.0},
        confidence=0.92,
    )

    payload = DiagnosisInput(
        incident_id="INC-001",
        service_id="payment-api",
        namespace="prod",
        environment="production",
        issue_summary="Memory usage is rising and pods restart.",
        signals={"memory_percent": 84.0, "restart_count": 5},
        recent_changes=["deploy v2.6", "config change on limits"],
        evidence=[evidence],
        constraints=["Do not execute a rollback automatically"],
    )

    assert payload.incident_id == "INC-001"
    assert payload.evidence[0].evidence_id == "EVD-METRIC-01"


def test_diagnosis_input_requires_nonempty_incident_id() -> None:
    with pytest.raises(ValidationError, match="incident_id"):
        DiagnosisInput(
            incident_id="",
            service_id="payment-api",
            namespace="prod",
            environment="production",
            issue_summary="Memory issue",
            signals={},
            evidence=[],
        )


def test_diagnosis_input_defaults_optional_collections() -> None:
    payload = DiagnosisInput(
        incident_id="INC-DEFAULTS",
        service_id="payment-api",
        namespace="test",
        environment="test",
        issue_summary="A minimal valid incident input.",
    )

    assert payload.signals == {}
    assert payload.recent_changes == []
    assert payload.evidence == []
    assert payload.constraints == []


@pytest.mark.parametrize(
    ("signal", "value"),
    [
        ("memory_percent", 101),
        ("cpu_percent", -1),
        ("restart_count", -1),
        ("latency_ms", "slow"),
        ("error_rate", float("nan")),
        ("cpu_percent", True),
    ],
)
def test_diagnosis_input_rejects_invalid_known_signals(signal: str, value: object) -> None:
    with pytest.raises(ValidationError, match=signal):
        DiagnosisInput(
            incident_id="INC-INVALID-SIGNAL",
            service_id="payment-api",
            namespace="test",
            environment="test",
            issue_summary="An invalid signal should fail validation.",
            signals={signal: value},
        )


def test_diagnosis_input_rejects_empty_change_entries() -> None:
    with pytest.raises(ValidationError, match="Recent changes"):
        DiagnosisInput(
            incident_id="INC-EMPTY-CHANGE",
            service_id="payment-api",
            namespace="test",
            environment="test",
            issue_summary="Recent change entry is empty.",
            recent_changes=[" "],
        )


def test_diagnosis_input_rejects_unknown_environment() -> None:
    with pytest.raises(ValidationError, match="environment"):
        DiagnosisInput(
            incident_id="INC-BAD-ENV",
            service_id="payment-api",
            namespace="test",
            environment="not-a-real-environment",
            issue_summary="The environment must use the shared enum values.",
        )


def test_diagnosis_input_rejects_duplicate_evidence_ids() -> None:
    evidence = EvidenceItem(
        evidence_id="EVD-DUPLICATE",
        evidence_type="metric",
        source="fixture://metrics",
        summary="Fixture measurement.",
        value={"cpu_percent": 80},
    )
    with pytest.raises(ValidationError, match="identifiers must be unique"):
        DiagnosisInput(
            incident_id="INC-DUPLICATE",
            service_id="payment-api",
            namespace="test",
            environment="test",
            issue_summary="Duplicate evidence identifiers are ambiguous.",
            evidence=[evidence, evidence],
        )
