from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]

from src.agents.diagnosis.contracts import DiagnosisAgentResult
from src.agents.diagnosis.envelope_builder import (
    DIAGNOSIS_EVENT_VERSION,
    DIAGNOSIS_SCHEMA_PATH,
    DiagnosisEnvelopeError,
    append_diagnosis_jsonl,
)
from src.agents.diagnosis.run_demo import run_demo
from src.agents.diagnosis.service import DiagnosisAgent
from src.contracts.enums import AgentId, ResultStatus
from src.contracts.events import EventEnvelope
from src.contracts.tasks import AgentTask, TaskContext

FIXTURES = Path(__file__).parent / "fixtures"


def _load_and_diagnose(fixture_name: str) -> tuple[EventEnvelope, DiagnosisAgentResult, bytes]:
    fixture_path = FIXTURES / fixture_name
    raw = fixture_path.read_bytes()
    received = EventEnvelope.model_validate(json.loads(raw))
    signals = received.payload["signals"]
    assert isinstance(signals, dict)
    incidents = signals["incidents"]
    assert isinstance(incidents, list) and len(incidents) == 1
    incident = incidents[0]
    assert isinstance(incident, dict)
    incident_id = incident["incident_id"]
    assert isinstance(incident_id, str)
    objective = f"Agent 1 anomaly for {received.service.service_id}."
    task = AgentTask(
        task_id=f"TEST-{received.message_id}",
        incident_id=incident_id,
        agent_id=AgentId.DIAGNOSIS,
        objective=objective,
        task_context=TaskContext(
            service_id=received.service.service_id,
            namespace=received.service.namespace,
            environment=received.environment,
            signals=signals,
            evidence_refs=received.evidence_refs,
        ),
        deadline=datetime.now(UTC) + timedelta(minutes=5),
    )
    result = asyncio.run(DiagnosisAgent().handle(task))
    assert isinstance(result, DiagnosisAgentResult)
    assert result.status is ResultStatus.SUCCEEDED
    assert result.incident_id == incident_id
    return received, result, raw


@pytest.mark.parametrize(
    "fixture_name",
    ["agent1_memory_oom.json", "agent1_memory_pressure.json"],
)
def test_fixture_generates_schema_valid_traced_envelope_without_mutation(
    fixture_name: str,
    tmp_path: Path,
) -> None:
    received, result, fixture_bytes = _load_and_diagnose(fixture_name)
    output_path = tmp_path / "diagnoses.jsonl"

    envelope, written = append_diagnosis_jsonl(received, result, output_path)

    assert written is True
    assert envelope.message_id != received.message_id
    assert envelope.topic == "diagnosis.completed"
    assert envelope.message_type == "DIAGNOSIS_COMPLETED"
    assert envelope.source == "diagnosis-agent"
    assert envelope.correlation_id == received.correlation_id
    assert envelope.trace_id == received.trace_id
    assert envelope.causation_id == received.message_id
    assert envelope.idempotency_key == (
        f"diagnosis-agent:diagnosis:{result.incident_id}:"
        f"{received.message_id}:{DIAGNOSIS_EVENT_VERSION}"
    )
    assert all(
        action.verification_profile == f"{received.service.service_id}-slo-default"
        for action in result.diagnosis.recommended_actions
    )
    assert any(
        received.correlation_id in limitation and "not resolved" in limitation
        for limitation in result.diagnosis.limitations
    )

    schema = json.loads(DIAGNOSIS_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(envelope.payload)
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    persisted = EventEnvelope.model_validate(json.loads(lines[0]))
    assert persisted == envelope
    assert (FIXTURES / fixture_name).read_bytes() == fixture_bytes


def test_replaying_same_message_does_not_append_duplicate_line(tmp_path: Path) -> None:
    received, result, _ = _load_and_diagnose("agent1_memory_oom.json")
    output_path = tmp_path / "diagnoses.jsonl"

    first, first_written = append_diagnosis_jsonl(received, result, output_path)
    second, second_written = append_diagnosis_jsonl(received, result, output_path)

    assert first_written is True
    assert second_written is False
    assert second.message_id == first.message_id
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 1


def test_different_messages_for_same_incident_append_two_lines(tmp_path: Path) -> None:
    first_received, first_result, _ = _load_and_diagnose("agent1_memory_oom.json")
    second_received, second_result, _ = _load_and_diagnose("agent1_memory_pressure.json")
    output_path = tmp_path / "diagnoses.jsonl"

    _, first_written = append_diagnosis_jsonl(first_received, first_result, output_path)
    _, second_written = append_diagnosis_jsonl(second_received, second_result, output_path)

    assert first_received.correlation_id == second_received.correlation_id
    assert first_received.message_id != second_received.message_id
    assert first_written is True
    assert second_written is True
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    events = [EventEnvelope.model_validate(json.loads(line)) for line in lines]
    assert len({event.idempotency_key for event in events}) == 2
    assert {event.causation_id for event in events} == {
        first_received.message_id,
        second_received.message_id,
    }
    diagnoses = [event.payload for event in events]
    assert {diagnosis["primary_root_cause"] for diagnosis in diagnoses} == {
        "MEMORY_LIMIT_REACHED",
        "MEMORY_GROWTH",
    }


def test_run_demo_processes_both_messages_and_reports_two_jsonl_lines(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "diagnoses.jsonl"

    assert run_demo(output_path) == 0

    output = capsys.readouterr().out
    assert "Diagnostic: MEMORY_LIMIT_REACHED" in output
    assert "Diagnostic: MEMORY_GROWTH" in output
    assert "JSONL_LINE_COUNT=2" in output
    events = [
        EventEnvelope.model_validate(json.loads(line))
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(events) == 2
    assert {event.causation_id for event in events} == {
        "88d4728a-b409-4162-ab8a-cd48f4846c35",
        "88d4728a-b409-4162-ab8a-cd48f4846c36",
    }


def test_received_message_without_message_id_is_rejected(tmp_path: Path) -> None:
    received, result, _ = _load_and_diagnose("agent1_memory_oom.json")
    received_data = received.model_dump(mode="json")
    received_data.pop("message_id")
    output_path = tmp_path / "missing-message-id.jsonl"

    with pytest.raises(DiagnosisEnvelopeError, match="non-empty message_id"):
        append_diagnosis_jsonl(received_data, result, output_path)

    assert not output_path.exists()


def test_invalid_shared_schema_refuses_write(tmp_path: Path) -> None:
    received, result, _ = _load_and_diagnose("agent1_memory_pressure.json")
    schema_path = tmp_path / "invalid.schema.json"
    schema_path.write_text('{"type": "string"}', encoding="utf-8")
    output_path = tmp_path / "not-written.jsonl"

    with pytest.raises(DiagnosisEnvelopeError, match="shared JSON Schema validation"):
        append_diagnosis_jsonl(
            received,
            result,
            output_path,
            schema_path=schema_path,
        )

    assert not output_path.exists()


def test_invalid_existing_jsonl_refuses_append(tmp_path: Path) -> None:
    received, result, _ = _load_and_diagnose("agent1_memory_oom.json")
    output_path = tmp_path / "malformed.jsonl"
    output_path.write_text("{not-json}\n", encoding="utf-8")
    original = output_path.read_bytes()

    with pytest.raises(DiagnosisEnvelopeError, match="Invalid JSONL event"):
        append_diagnosis_jsonl(received, result, output_path)

    assert output_path.read_bytes() == original
