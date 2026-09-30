"""Source-backed Agent 6 assessment and local file reader boundaries."""

import asyncio
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from src.agents.mlops_lifecycle.adapters.local_files import (
    FileBaselineRepository,
    FileLabelRepository,
    FilePredictionRepository,
    LocalSourceConfig,
)
from src.agents.mlops_lifecycle.service import MLOpsLifecycleAgent
from src.contracts.mlops_observations import ModelWindowQuery, PredictionObservation
from src.platform.errors import IntegrationNotConfiguredError

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "contracts/examples/agent6/local-sources"


@pytest.fixture
def source_config() -> LocalSourceConfig:
    return LocalSourceConfig.model_validate_json(
        (SOURCES / "machine-sources.json").read_text(encoding="utf-8")
    )


def make_agent(
    baseline: Path = SOURCES / "machine-baseline.json",
    predictions: Path = SOURCES / "machine-predictions.json",
    labels: Path = SOURCES / "machine-labels.json",
) -> MLOpsLifecycleAgent:
    return MLOpsLifecycleAgent(
        baselines=FileBaselineRepository(baseline),
        predictions=FilePredictionRepository(predictions),
        labels=FileLabelRepository(labels),
    )


def test_sources_complete_machine_model_assessment_with_hashed_provenance(source_config):
    report = asyncio.run(make_agent().assess_model_window(source_config.query))
    assert report.assessment.model_name == "machine-failure"
    assert report.assessment.model_version == "v1"
    assert report.assessment.decision == "RETRAIN_AND_EVALUATE"
    assert report.performance.labeled_samples == 60
    assert report.performance.current == pytest.approx(2 / 3)
    assert {item.name: item.current_samples for item in report.distributions} == {
        "temperature_c": 100,
        "failure_probability": 100,
        "machine_type": 100,
    }
    assert {item.name: item.detected for item in report.distributions} == {
        "temperature_c": True,
        "failure_probability": True,
        "machine_type": False,
    }
    expected_hashes = tuple(
        "sha256:" + sha256((SOURCES / name).read_bytes()).hexdigest()
        for name in ("machine-baseline.json", "machine-predictions.json", "machine-labels.json")
    )
    assert report.assessment.evidence_refs == expected_hashes
    assert all("machine-predictions.json" not in ref for ref in expected_hashes)
    assert "observation_id" not in report.model_dump_json()


def test_label_cutoff_changes_coverage_without_changing_prediction_window(source_config):
    query = source_config.query
    early = asyncio.run(make_agent().assess_model_window(query))
    later = query.model_copy(update={"as_of": datetime(2026, 9, 30, 10, tzinfo=UTC)})
    updated = asyncio.run(make_agent().assess_model_window(later))
    assert early.performance.labeled_samples == 60
    assert updated.performance.labeled_samples == 61
    assert early.distributions == updated.distributions


def test_unconfigured_agent_rejects_source_assessment(source_config):
    with pytest.raises(IntegrationNotConfiguredError, match="data readers"):
        asyncio.run(MLOpsLifecycleAgent().assess_model_window(source_config.query))


def test_wrong_baseline_model_is_rejected():
    with pytest.raises(ValueError, match="Baseline model identity"):
        asyncio.run(
            FileBaselineRepository(SOURCES / "machine-baseline.json").get("another-model", "v1")
        )


def test_file_readers_select_model_window_and_confirmed_labels(source_config):
    query = source_config.query
    batch = asyncio.run(
        FilePredictionRepository(SOURCES / "machine-predictions.json").list_window(
            query.model_name, query.model_version, query.window_start, query.window_end
        )
    )
    assert len(batch.records) == 100
    label_batch = asyncio.run(
        FileLabelRepository(SOURCES / "machine-labels.json").list_confirmed(
            tuple(record.observation_id for record in batch.records), query.as_of
        )
    )
    assert len(label_batch.records) == 60


def test_empty_model_window_fails_explicitly(source_config):
    body = source_config.query.model_dump(mode="json")
    body["window_start"] = "2026-09-29T10:00:00Z"
    body["window_end"] = "2026-09-29T11:00:00Z"
    body["as_of"] = "2026-09-29T11:00:00Z"
    with pytest.raises(ValueError, match="No predictions"):
        asyncio.run(make_agent().assess_model_window(ModelWindowQuery.model_validate(body)))


def test_invalid_source_file_is_rejected_without_echoing_records(tmp_path, source_config):
    bad = tmp_path / "predictions.json"
    bad.write_text('[{"private_feature":"secret"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid prediction-record JSON") as error:
        asyncio.run(make_agent(predictions=bad).assess_model_window(source_config.query))
    assert "secret" not in str(error.value)


def test_all_labels_missing_asks_for_labels(tmp_path, source_config):
    empty = tmp_path / "labels.json"
    empty.write_text("[]", encoding="utf-8")
    report = asyncio.run(make_agent(labels=empty).assess_model_window(source_config.query))
    assert report.assessment.decision == "COLLECT_LABELS"
    assert report.performance.labeled_samples == 0
    assert len(report.assessment.evidence_refs) == 2


def test_file_digest_changes_when_local_source_changes(tmp_path):
    source = SOURCES / "machine-labels.json"
    original = source.read_bytes()
    modified = tmp_path / "labels.json"
    modified.write_bytes(original + b" ")
    reader = FileLabelRepository(modified)
    batch = asyncio.run(reader.list_confirmed(("M-001",), datetime(2026, 9, 29, 10, tzinfo=UTC)))
    assert batch.evidence_ref == "sha256:" + sha256(original + b" ").hexdigest()
    assert batch.evidence_ref != "sha256:" + sha256(original).hexdigest()


def test_legacy_fraud_field_names_remain_readable_but_serialize_generically():
    body = json.loads(
        (ROOT / "contracts/examples/agent6/observation-window.json").read_text(encoding="utf-8")
    )["predictions"][0]
    parsed = PredictionObservation.model_validate(body)
    assert parsed.observation_id == body["transaction_id"]
    assert parsed.positive_class_score == body["fraud_score"]
    serialized = parsed.model_dump(mode="json")
    assert "observation_id" in serialized and "positive_class_score" in serialized
    assert "transaction_id" not in serialized and "fraud_score" not in serialized
