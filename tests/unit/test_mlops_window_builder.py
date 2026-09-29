"""Meaningful boundaries for converting transaction records into a health assessment."""

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agents.mlops_lifecycle.health import RuleBasedModelHealthAssessor
from src.agents.mlops_lifecycle.window_builder import WindowBuildError, build_model_health_input
from src.contracts.mlops_observations import WindowBuildRequest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "contracts/examples/agent6/observation-window.json"


@pytest.fixture
def payload():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def build(payload):
    return build_model_health_input(WindowBuildRequest.model_validate(payload))


def test_records_build_three_histograms_and_aligned_delayed_labels(payload):
    health = build(payload)
    counts = {item.name: item.current_counts for item in health.distributions}
    assert counts == {
        "transaction_amount": (5, 15, 30, 50),
        "fraud_probability": (20, 30, 50),
        "merchant_category": (40, 35, 20, 5),
    }
    assert health.distributions[0].bins == ("lt-50", "50-100", "100-500", "gte-500")
    assert health.performance.labels[:20] == (1,) * 20
    assert health.performance.labels[20:60] == (0,) * 40
    assert health.performance.labels[60:] == (None,) * 40
    assert len(health.evidence_refs) == 3
    assert "transaction_id" not in health.model_dump_json()
    report = asyncio.run(RuleBasedModelHealthAssessor().assess(health))
    assert report.assessment.decision == "RETRAIN_AND_EVALUATE"
    assert report.performance.labeled_samples == 60


def test_numeric_bins_are_lower_inclusive_at_each_edge(payload):
    selected = payload["predictions"][:4]
    for record, value in zip(selected, [49.99, 50, 100, 500], strict=True):
        record["features"]["transaction_amount"] = value
    payload["predictions"] = selected
    payload["labels"] = []
    health = build(payload)
    assert health.distributions[0].current_counts == (1, 1, 1, 1)
    assert health.performance.labels == (None,) * 4
    assert len(health.evidence_refs) == 2


def test_window_end_is_exclusive(payload):
    outside = dict(payload["predictions"][0])
    outside["transaction_id"] = "TX-outside"
    outside["predicted_at"] = payload["window_end"]
    payload["predictions"].append(outside)
    assert sum(build(payload).distributions[0].current_counts) == 100


def test_label_becomes_available_after_as_of_advances(payload):
    first = build(payload)
    assert first.performance.labels[60] is None
    payload["as_of"] = "2026-09-30T10:00:00Z"
    later = build(payload)
    assert later.performance.labels[60] == 0


@pytest.mark.parametrize(
    "problem,message",
    [
        ("duplicate_prediction", "Duplicate prediction"),
        ("wrong_model", "model identity"),
        ("duplicate_label", "Duplicate confirmed label"),
        ("early_label", "predates its prediction"),
        ("missing_label_evidence", "labels_evidence_ref"),
        ("missing_numeric", "Missing value for numeric"),
        ("numeric_text", "finite number"),
    ],
)
def test_ambiguous_or_invalid_records_fail_closed(payload, problem, message):
    if problem == "duplicate_prediction":
        payload["predictions"].append(payload["predictions"][0])
    elif problem == "wrong_model":
        payload["predictions"][0]["model_version"] = "other-version"
    elif problem == "duplicate_label":
        payload["labels"].append(payload["labels"][0])
    elif problem == "early_label":
        payload["labels"][0]["confirmed_at"] = "2026-09-29T07:00:00Z"
    elif problem == "missing_label_evidence":
        payload["labels_evidence_ref"] = None
    elif problem == "missing_numeric":
        payload["predictions"][0]["features"].pop("transaction_amount")
    else:
        payload["predictions"][0]["features"]["transaction_amount"] = "not-a-number"
    with pytest.raises(WindowBuildError, match=message):
        build(payload)


def test_empty_selected_window_is_explicit_error(payload):
    payload["window_start"] = "2026-09-29T11:00:00Z"
    payload["window_end"] = "2026-09-29T12:00:00Z"
    payload["as_of"] = "2026-09-29T12:00:00Z"
    with pytest.raises(WindowBuildError, match="No predictions"):
        build(payload)


@pytest.mark.parametrize(
    "problem",
    [
        "unsorted_edges",
        "duplicate_bin",
        "wrong_reference_length",
        "duplicate_distribution",
        "as_of_before_end",
        "boolean_feature",
        "nan_feature",
        "invalid_label",
    ],
)
def test_invalid_baseline_or_records_rejected_by_contract(payload, problem):
    distributions = payload["baseline"]["distributions"]
    if problem == "unsorted_edges":
        distributions[0]["spec"]["edges"] = [50, 50, 500]
    elif problem == "duplicate_bin":
        distributions[0]["spec"]["bins"][1] = "lt-50"
    elif problem == "wrong_reference_length":
        distributions[0]["reference_counts"] = [25, 25]
    elif problem == "duplicate_distribution":
        distributions.append(distributions[0])
    elif problem == "as_of_before_end":
        payload["as_of"] = payload["window_start"]
    elif problem == "boolean_feature":
        payload["predictions"][0]["features"]["transaction_amount"] = True
    elif problem == "nan_feature":
        payload["predictions"][0]["features"]["transaction_amount"] = float("nan")
    else:
        payload["labels"][0]["label"] = 2
    with pytest.raises(ValidationError):
        WindowBuildRequest.model_validate(payload)


def test_all_labels_missing_returns_collect_labels(payload):
    payload["labels"] = []
    payload["labels_evidence_ref"] = None
    health = build(payload)
    report = asyncio.run(RuleBasedModelHealthAssessor().assess(health))
    assert report.assessment.decision == "COLLECT_LABELS"
    assert report.performance.ground_truth_coverage == 0
    assert report.performance.current is None
