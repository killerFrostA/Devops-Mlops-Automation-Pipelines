import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from pydantic import ValidationError

from src.contracts.events import EventEnvelope
from src.contracts.topics import TOPICS

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = sorted((ROOT / "contracts/examples/v1").glob("*.json"))
pytestmark = pytest.mark.contract


def validator() -> Draft202012Validator:
    schema = json.loads((ROOT / "contracts/jsonschema/v1/EventEnvelope.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.stem)
def test_all_event_examples_match_python_and_jsonschema(path: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    event = EventEnvelope.model_validate(document)
    validator().validate(document)
    assert event.partition_key == event.correlation_id


def test_every_topic_has_an_example() -> None:
    assert {path.stem for path in EXAMPLES} == set(TOPICS)


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": "2.0"},
        {"confidence": 1.1},
        {"confidence": float("nan")},
        {"topic": "unknown.topic"},
        {"message_type": "ACTION_COMPLETED"},
        {"unexpected": "not in contract"},
        {"timestamp": "2026-09-29T12:00:00"},
        {"payload": {"anomaly_id": "missing-fields"}},
    ],
)
def test_python_rejects_malformed_event(anomaly: EventEnvelope, changes: dict) -> None:
    document = anomaly.model_dump(mode="json")
    document.update(changes)
    with pytest.raises(ValidationError):
        EventEnvelope.model_validate(document)


def test_schema_rejects_wrong_topic_payload_and_event_fields(anomaly: EventEnvelope) -> None:
    for changes in (
        {"message_type": "ACTION_COMPLETED"},
        {"payload": {"anomaly_id": "missing-fields"}},
        {"topic": "unknown.topic"},
        {"unexpected": "invalid"},
    ):
        document = copy.deepcopy(anomaly.model_dump(mode="json"))
        document.update(changes)
        assert list(validator().iter_errors(document))


def test_semantic_deadline_order_is_enforced(anomaly: EventEnvelope) -> None:
    document = anomaly.model_dump(mode="json")
    document["deadline"] = "2026-09-29T11:59:59Z"
    with pytest.raises(ValidationError, match="deadline"):
        EventEnvelope.model_validate(document)


def test_exported_schemas_are_current() -> None:
    subprocess.run([sys.executable, "scripts/export_contracts.py", "--check"], cwd=ROOT, check=True)
