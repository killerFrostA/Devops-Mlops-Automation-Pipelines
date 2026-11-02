"""Build and persist validated diagnosis.completed event envelopes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError, ValidationError as JsonSchemaValidationError
from pydantic import ValidationError

from src.agents.diagnosis.contracts import DiagnosisAgentResult
from src.contracts.events import EventEnvelope


DIAGNOSIS_EVENT_VERSION = "1"
DIAGNOSIS_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "contracts"
    / "jsonschema"
    / "v1"
    / "DiagnosisCompleted.schema.json"
)


class DiagnosisEnvelopeError(ValueError):
    """Raised when a diagnosis event cannot be validated or safely persisted."""


def _require_received_message(received: EventEnvelope | dict[str, Any]) -> EventEnvelope:
    if isinstance(received, dict):
        message_id = received.get("message_id")
        if not isinstance(message_id, str) or not message_id.strip():
            raise DiagnosisEnvelopeError(
                "Received event must include a non-empty message_id for idempotency."
            )
        try:
            return EventEnvelope.model_validate(received)
        except ValidationError as exc:
            raise DiagnosisEnvelopeError(f"Invalid received event envelope: {exc}") from exc
    if not isinstance(received.message_id, str) or not received.message_id.strip():
        raise DiagnosisEnvelopeError(
            "Received event must include a non-empty message_id for idempotency."
        )
    return received


def build_diagnosis_envelope(
    received: EventEnvelope | dict[str, Any],
    result: DiagnosisAgentResult,
) -> EventEnvelope:
    """Create a shared diagnosis.completed envelope correlated to the input message."""
    received = _require_received_message(received)
    evidence_refs = tuple(
        dict.fromkeys(
            reference
            for hypothesis in result.diagnosis.hypotheses
            for reference in hypothesis.evidence_refs
        )
    )
    return EventEnvelope(
        # One Agent 1 message can carry several incidents; each diagnosis keeps its own ID.
        correlation_id=result.incident_id,
        causation_id=received.message_id,
        trace_id=received.trace_id,
        source="diagnosis-agent",
        target="orchestrator",
        topic="diagnosis.completed",
        message_type="DIAGNOSIS_COMPLETED",
        priority=received.priority,
        severity=received.severity,
        environment=received.environment,
        service=received.service,
        evidence_refs=evidence_refs,
        confidence=result.confidence,
        requires_ack=True,
        idempotency_key=(
            f"diagnosis-agent:diagnosis:{result.incident_id}:"
            f"{received.message_id}:{DIAGNOSIS_EVENT_VERSION}"
        ),
        payload=result.diagnosis.model_dump(mode="json"),
    )


def _validate_diagnosis_payload(payload: dict[str, Any], schema_path: Path) -> None:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(payload)
    except (OSError, json.JSONDecodeError, SchemaError, JsonSchemaValidationError) as exc:
        raise DiagnosisEnvelopeError(
            f"DiagnosisCompleted payload failed shared JSON Schema validation "
            f"('{schema_path}'): {exc}"
        ) from exc


def append_diagnosis_jsonl(
    received: EventEnvelope | dict[str, Any],
    result: DiagnosisAgentResult,
    output_path: Path,
    *,
    schema_path: Path = DIAGNOSIS_SCHEMA_PATH,
) -> tuple[EventEnvelope, bool]:
    """Validate and append an envelope unless its idempotency key is already present."""
    received = _require_received_message(received)
    envelope = build_diagnosis_envelope(received, result)
    payload = envelope.payload
    _validate_diagnosis_payload(payload, schema_path)

    output_path = Path(output_path)
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        existing_data = json.loads(line)
                        existing = EventEnvelope.model_validate(existing_data)
                    except (json.JSONDecodeError, ValidationError) as exc:
                        raise DiagnosisEnvelopeError(
                            f"Invalid JSONL event at '{output_path}' line {line_number}: {exc}"
                        ) from exc
                    if existing.idempotency_key == envelope.idempotency_key:
                        return existing, False
        except OSError as exc:
            raise DiagnosisEnvelopeError(
                f"Cannot read diagnosis JSONL output '{output_path}': {exc}"
            ) from exc

    serialized = json.dumps(envelope.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized + "\n")
            stream.flush()
    except OSError as exc:
        raise DiagnosisEnvelopeError(
            f"Cannot write diagnosis JSONL output '{output_path}': {exc}"
        ) from exc
    return envelope, True
