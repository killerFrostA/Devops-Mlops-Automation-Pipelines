"""Replay the local Agent 1 fixtures and append diagnosis events to JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.agents.diagnosis.agent1_runner import Agent1InputError, diagnose_agent1_event
from src.agents.diagnosis.contracts import DiagnosisAgentResult
from src.agents.diagnosis.envelope_builder import (
    DiagnosisEnvelopeError,
    append_diagnosis_jsonl,
    build_diagnosis_envelope,
)
from src.contracts.events import EventEnvelope

FIXTURES = Path(__file__).resolve().parent / "tests" / "fixtures"
DEMO_FIXTURES = ("agent1_memory_oom.json", "agent1_memory_pressure.json")


def run_demo(output_path: Path) -> int:
    for fixture_name in DEMO_FIXTURES:
        fixture_path = FIXTURES / fixture_name
        try:
            received = EventEnvelope.model_validate(
                json.loads(fixture_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise DiagnosisEnvelopeError(
                f"Cannot load Agent 1 fixture '{fixture_path}': {exc}"
            ) from exc

        for diagnosis in diagnose_agent1_event(received):
            _print_diagnosis(fixture_name, received, diagnosis.result, output_path)
    line_count = len(output_path.read_text(encoding="utf-8").splitlines())
    print(f"JSONL_LINE_COUNT={line_count}")
    return 0


def _print_diagnosis(
    fixture_name: str,
    received: EventEnvelope,
    result: DiagnosisAgentResult,
    output_path: Path,
) -> None:
    envelope = build_diagnosis_envelope(received, result)
    persisted, written = append_diagnosis_jsonl(received, result, output_path)

    print(f"Fixture: {fixture_name}")
    print(f"Diagnostic: {result.diagnosis.primary_root_cause}")
    print("Hypotheses:")
    for hypothesis in result.diagnosis.hypotheses:
        print(
            f"- {hypothesis.code} ({hypothesis.probability:.2f}); "
            f"evidence_refs={list(hypothesis.evidence_refs)}"
        )
    print("Evidence:")
    for item in result.evidence:
        print(f"- {item.evidence_id}: {item.summary}")
    print("Recommended actions:")
    for action in result.diagnosis.recommended_actions:
        print(f"- {action.action.value}")
    if not result.diagnosis.recommended_actions:
        print("- none")
    print("Limitations:")
    for limitation in result.diagnosis.limitations:
        print(f"- {limitation}")
    print(f"JSONL: {'appended' if written else 'duplicate skipped'} ({output_path})")
    print(
        "Persisted diagnosis envelope:"
        if written
        else "Diagnosis envelope candidate (duplicate key; not persisted):"
    )
    print(json.dumps(envelope.model_dump(mode="json"), ensure_ascii=False, indent=2))
    if not written:
        print(f"Existing idempotent record: {persisted.message_id}")
        print("Existing persisted envelope:")
        print(json.dumps(persisted.model_dump(mode="json"), ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay Agent 1 diagnosis fixtures and append diagnosis.completed JSONL events."
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        type=Path,
        help="Output JSONL path; no Kafka connection is used.",
    )
    args = parser.parse_args(argv)
    try:
        return run_demo(args.output_jsonl)
    except (Agent1InputError, DiagnosisEnvelopeError) as exc:
        print(f"Diagnosis demo error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
