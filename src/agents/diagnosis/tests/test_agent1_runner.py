import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from src.agents.diagnosis.agent1_runner import (
    Agent1InputError,
    diagnose_agent1_event,
    load_agent1_events,
    run_agent1_file,
)
from src.contracts.enums import ActionType
from src.contracts.events import EventEnvelope

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict[str, Any]:
    fixture: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return fixture


def _write_jsonl(path: Path, *messages: dict[str, Any]) -> Path:
    path.write_text("".join(json.dumps(message) + "\n" for message in messages), encoding="utf-8")
    return path


def test_architecture_format_with_dated_deployment_recommends_rollback() -> None:
    received = EventEnvelope.model_validate(_fixture("agent1_architecture_example.json"))

    [diagnosis] = diagnose_agent1_event(received)
    result = diagnosis.result

    assert result.incident_id == "INC-2026-0087"
    assert result.diagnosis.primary_root_cause == "MEMORY_LEAK_POST_DEPLOY"
    actions = result.diagnosis.recommended_actions
    assert actions[0].action is ActionType.ROLLBACK
    assert actions[0].parameters["deployment_id"] == "DEP-773"
    assert ActionType.SCALE_REPLICAS in [action.action for action in actions]
    assert result.confidence == result.diagnosis.hypotheses[0].probability
    assert any(
        item.evidence_type == "git_change"
        and item.evidence_id in result.diagnosis.hypotheses[0].evidence_refs
        for item in result.evidence
    )
    # Past incident with the same root cause is ranked first by the local RAG.
    assert result.knowledge.similar_incidents[0].reference_id == "LOCAL-INC-120"
    assert (
        "LOCAL-INC-120 (MEMORY_LEAK_POST_DEPLOY, same cause)" in result.diagnosis.rationale_summary
    )


def test_architecture_format_without_change_context_does_not_recommend_rollback() -> None:
    message = _fixture("agent1_architecture_example.json")
    del message["payload"]["signals"]["change_context"]

    [diagnosis] = diagnose_agent1_event(EventEnvelope.model_validate(message))

    assert diagnosis.result.diagnosis.primary_root_cause == "MEMORY_PRESSURE"
    assert ActionType.ROLLBACK not in [
        action.action for action in diagnosis.result.diagnosis.recommended_actions
    ]


def test_deployment_too_close_to_detection_is_not_a_post_deploy_leak() -> None:
    message = _fixture("agent1_architecture_example.json")
    message["payload"]["signals"]["change_context"]["minutes_since_deployment"] = 2

    [diagnosis] = diagnose_agent1_event(EventEnvelope.model_validate(message))

    assert diagnosis.result.diagnosis.primary_root_cause == "MEMORY_PRESSURE"


def test_agent1_oom_message_recommends_memory_limit_and_ranks_related_incident() -> None:
    [diagnosis] = diagnose_agent1_event(
        EventEnvelope.model_validate(_fixture("agent1_memory_oom.json"))
    )
    result = diagnosis.result

    assert result.diagnosis.primary_root_cause == "MEMORY_LIMIT_REACHED"
    assert [item.action for item in result.explainability.recommendation_explanations] == [
        "INCREASE_MEMORY_LIMIT",
        "COLLECT_MORE_EVIDENCE",
    ]
    # No shared ActionType exists for a memory-limit change; it is reported, not invented.
    assert result.diagnosis.recommended_actions == ()
    assert any(
        "INCREASE_MEMORY_LIMIT" in item and "no equivalent" in item
        for item in result.diagnosis.limitations
    )
    causes = [item.root_cause for item in result.knowledge.similar_incidents]
    assert causes[0] == "MEMORY_LIMIT_REACHED"
    assert "INSUFFICIENT_MEMORY_LIMIT" in causes


def test_message_with_two_incidents_produces_one_diagnosis_per_incident() -> None:
    message = _fixture("agent1_memory_oom.json")
    second = deepcopy(message["payload"]["signals"]["incidents"][0])
    second["incident_id"] = "INC-2026-second"
    second["service_id"] = "payments-db"
    message["payload"]["signals"]["incidents"].append(second)
    message["payload"]["signals"]["categories"] = ["memory_oom"]

    diagnoses = diagnose_agent1_event(EventEnvelope.model_validate(message))

    assert [item.result.incident_id for item in diagnoses] == [
        "INC-2026-25bc2c7bae",
        "INC-2026-second",
    ]
    assert diagnoses[1].task.task_context.service_id == "payments-db"
    assert all(
        isinstance(incidents := item.task.task_context.signals["incidents"], list)
        and len(incidents) == 1
        for item in diagnoses
    )


def test_jsonl_file_is_diagnosed_reported_and_idempotent(tmp_path: Path) -> None:
    input_path = _write_jsonl(
        tmp_path / "agent1.jsonl",
        _fixture("agent1_memory_oom.json"),
        _fixture("agent1_memory_pressure.json"),
    )
    output_dir = tmp_path / "out"

    written = run_agent1_file(input_path, output_dir)

    assert [item.diagnosis.result.diagnosis.primary_root_cause for item in written] == [
        "MEMORY_LIMIT_REACHED",
        "MEMORY_GROWTH",
    ]
    assert all(item.jsonl_written for item in written)
    assert all(item.json_path.exists() and item.markdown_path.exists() for item in written)
    report = written[0].markdown_path.read_text(encoding="utf-8")
    assert "Incidents passés similaires" in report
    assert "LOCAL-INC-122" in report
    events = (output_dir / "diagnosis.completed.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(events) == 2
    first_event = json.loads(events[0])
    assert first_event["topic"] == "diagnosis.completed"
    assert first_event["causation_id"] == "88d4728a-b409-4162-ab8a-cd48f4846c35"
    assert first_event["confidence"] == 0.8

    replayed = run_agent1_file(input_path, output_dir)

    assert not any(item.jsonl_written for item in replayed)
    assert (
        len((output_dir / "diagnosis.completed.jsonl").read_text(encoding="utf-8").splitlines())
        == 2
    )


def test_json_array_input_is_accepted(tmp_path: Path) -> None:
    input_path = tmp_path / "agent1.json"
    input_path.write_text(
        json.dumps(
            [_fixture("agent1_memory_oom.json"), _fixture("agent1_architecture_example.json")]
        ),
        encoding="utf-8",
    )

    assert [event.correlation_id for event in load_agent1_events(input_path)] == [
        "INC-2026-25bc2c7bae",
        "INC-2026-0087",
    ]


def test_non_anomaly_messages_and_invalid_lines_are_rejected(tmp_path: Path) -> None:
    shared_examples = Path(__file__).resolve().parents[4] / "contracts" / "examples" / "v1"
    other = json.loads((shared_examples / "diagnosis.completed.json").read_text(encoding="utf-8"))
    with pytest.raises(Agent1InputError, match="expected monitoring.anomaly.detected"):
        load_agent1_events(_write_jsonl(tmp_path / "other.jsonl", other))

    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"message_id": "x"}\n{not json}\n', encoding="utf-8")
    with pytest.raises(Agent1InputError, match="line 2"):
        load_agent1_events(broken)

    empty = tmp_path / "empty.jsonl"
    empty.write_text("  \n", encoding="utf-8")
    with pytest.raises(Agent1InputError, match="empty"):
        load_agent1_events(empty)
