import json
from datetime import UTC, datetime
from pathlib import Path

import jsonschema  # type: ignore[import-untyped]

from src.agents.diagnosis.contracts import DiagnosisInput
from src.agents.diagnosis.evidence_collector import collect_evidence
from src.agents.diagnosis.result_builder import build_result, to_diagnosis_completed
from src.contracts.enums import AgentId, Environment
from src.contracts.payloads import DiagnosisCompleted
from src.contracts.tasks import AgentTask, TaskContext

SCHEMA_PATH = (
    Path(__file__).resolve().parents[4]
    / "contracts"
    / "jsonschema"
    / "v1"
    / "DiagnosisCompleted.schema.json"
)


def _task() -> AgentTask:
    return AgentTask(
        task_id="TASK-001",
        incident_id="INC-TASK",
        agent_id=AgentId.DIAGNOSIS,
        objective="Diagnose the incident",
        task_context=TaskContext(
            service_id="payment-api",
            namespace="prod",
            environment=Environment.PRODUCTION,
            constraints=("Do not execute actions automatically.",),
        ),
        deadline=datetime.now(UTC),
    )


def _validate_schema(diagnosis: DiagnosisCompleted) -> None:
    with SCHEMA_PATH.open(encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    jsonschema.validate(
        instance=diagnosis.model_dump(mode="json"),
        schema=schema,
    )


def test_hypothesis_code_is_a_free_string_in_shared_schema() -> None:
    with SCHEMA_PATH.open(encoding="utf-8") as schema_file:
        schema = json.load(schema_file)

    code_schema = schema["$defs"]["Hypothesis"]["properties"]["code"]
    assert code_schema["type"] == "string"
    assert "enum" not in code_schema


def test_conversion_keeps_schema_and_withholds_unproven_rollback() -> None:
    payload = DiagnosisInput(
        incident_id="INC-FROM-PAYLOAD",
        service_id="payment-api",
        namespace="prod",
        environment=Environment.PRODUCTION,
        issue_summary="Memory is high after a deployment.",
        signals={"memory_pct": 84, "restart_count": 4},
        recent_changes=["deploy v2.6"],
    )

    diagnosis = to_diagnosis_completed(_task(), build_result(payload))

    _validate_schema(diagnosis)
    assert diagnosis.diagnosis_id == "DIAG-INC-TASK"
    assert diagnosis.recommended_actions == ()
    assert diagnosis.requires_additional_evidence is True
    assert any("complete dated evidence" in item.lower() for item in diagnosis.limitations)
    evidence_ids = {
        evidence_id
        for hypothesis in diagnosis.hypotheses
        for evidence_id in hypothesis.evidence_refs
    }
    result_evidence_ids = {item.evidence_id for item in collect_evidence(payload)}
    assert evidence_ids <= result_evidence_ids


def test_insufficient_evidence_conversion_has_no_action_and_valid_schema() -> None:
    payload = DiagnosisInput(
        incident_id="INC-FROM-PAYLOAD",
        service_id="payment-api",
        namespace="prod",
        environment=Environment.PRODUCTION,
        issue_summary="The cause is unknown.",
    )

    diagnosis = to_diagnosis_completed(_task(), build_result(payload))

    _validate_schema(diagnosis)
    assert diagnosis.recommended_actions == ()
    assert diagnosis.requires_additional_evidence is True
    assert diagnosis.limitations
    assert diagnosis.hypotheses[0].evidence_refs == ()


def test_error_rate_signal_can_rank_resource_pressure() -> None:
    payload = DiagnosisInput(
        incident_id="INC-003",
        service_id="payment-api",
        namespace="prod",
        environment=Environment.PRODUCTION,
        issue_summary="Requests are failing.",
        signals={"error_rate": 1.0},
    )

    result = build_result(payload)

    assert result.primary_root_cause == "RESOURCE_PRESSURE_OR_CPU_STARVATION"
    assert result.hypotheses[0].evidence_refs == ("EVD-INC-003-METRIC-05",)
    assert [item.action for item in result.recommended_actions] == ["COLLECT_MORE_EVIDENCE"]


def test_resource_recommendation_maps_to_shared_scale_action() -> None:
    payload = DiagnosisInput(
        incident_id="INC-SCALE",
        service_id="payment-api",
        namespace="prod",
        environment=Environment.PRODUCTION,
        issue_summary="CPU pressure is causing long request latency.",
        signals={"cpu_percent": 95, "latency_ms": 900},
    )

    diagnosis = to_diagnosis_completed(_task(), build_result(payload))

    assert diagnosis.primary_root_cause == "RESOURCE_PRESSURE_OR_CPU_STARVATION"
    assert len(diagnosis.recommended_actions) == 1
    assert diagnosis.recommended_actions[0].action.value == "SCALE_REPLICAS"
    assert any("does not specify a replica count" in item for item in diagnosis.limitations)
