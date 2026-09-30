import asyncio
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agents.mlops_lifecycle.health.assessor import (
    InsufficientSamplesError,
    RuleBasedModelHealthAssessor,
)
from src.agents.mlops_lifecycle.health.metrics import (
    binary_confusion_counts,
    population_stability_index,
)
from src.agents.mlops_lifecycle.service import MLOpsLifecycleAgent
from src.contracts.enums import AgentId, ResultStatus
from src.contracts.mlops import (
    DistributionComparison,
    LabeledPredictionWindow,
    ModelHealthInput,
    ModelHealthPolicy,
)
from src.contracts.payloads import ModelHealthAssessed
from src.contracts.tasks import AgentResult, AgentTask, TaskContext

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def health_input() -> ModelHealthInput:
    return ModelHealthInput.model_validate_json(
        (ROOT / "contracts/examples/agent6/model-health-input.json").read_text(encoding="utf-8")
    )


def assess(request: ModelHealthInput, policy: ModelHealthPolicy | None = None):
    return asyncio.run(RuleBasedModelHealthAssessor(policy).assess(request))


def with_performance(request, predictions, labels, baseline=0.9):
    body = request.model_dump(mode="json")
    body["distributions"] = [body["distributions"][0]]
    body["performance"] = {
        "predictions": predictions,
        "labels": labels,
        "baseline_f1": baseline,
    }
    return ModelHealthInput.model_validate(body)


def make_task(request: ModelHealthInput) -> AgentTask:
    return AgentTask(
        task_id="TASK-health",
        incident_id="INC-health",
        agent_id=AgentId.MLOPS_LIFECYCLE,
        objective="assess_model_health",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
        task_context=TaskContext(
            service_id="fraud-api",
            namespace="test",
            environment="test",
            signals={"model_health": request.model_dump(mode="json")},
            evidence_refs=request.evidence_refs,
        ),
    )


def distribution(reference, current):
    return DistributionComparison(
        name="amount",
        scope="feature",
        bins=("low", "high"),
        reference_counts=reference,
        current_counts=current,
    )


def test_psi_zero_for_identical_proportions_and_scaled_counts():
    assert population_stability_index(distribution((20, 80), (200, 800))) == 0


def test_psi_matches_independent_hand_calculation():
    epsilon = 1e-6
    p = (0.5 + epsilon) / (1 + 2 * epsilon)
    q = (0.9 + epsilon) / (1 + 2 * epsilon)
    other_q = (0.1 + epsilon) / (1 + 2 * epsilon)
    expected = (q - p) * math.log(q / p) + (other_q - p) * math.log(other_q / p)
    assert population_stability_index(distribution((50, 50), (90, 10))) == pytest.approx(expected)


def test_psi_handles_zero_bins_and_is_symmetric():
    left = population_stability_index(distribution((100, 0), (0, 100)))
    right = population_stability_index(distribution((0, 100), (100, 0)))
    assert math.isfinite(left) and left > 20
    assert left == pytest.approx(right)


@pytest.mark.parametrize("epsilon", [0, -1, 0.1, float("nan"), float("inf")])
def test_invalid_smoothing_rejected(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        population_stability_index(distribution((50, 50), (50, 50)), epsilon)


@pytest.mark.parametrize(
    "reference,current",
    [
        ((0, 0), (50, 50)),
        ((-1, 101), (50, 50)),
        ((1.0, 99), (50, 50)),
        ((True, 99), (50, 50)),
        ((50, 50), (100,)),
        ((50, 50), (50, 50, 0)),
    ],
)
def test_invalid_histograms_rejected(reference, current):
    with pytest.raises(ValidationError):
        distribution(reference, current)


def test_duplicate_bins_rejected():
    with pytest.raises(ValidationError, match="unique"):
        DistributionComparison(
            name="amount",
            scope="feature",
            bins=("same", "same"),
            reference_counts=(50, 50),
            current_counts=(50, 50),
        )


@pytest.mark.parametrize(
    "predictions,labels",
    [
        ([1], []),
        ([2], [1]),
        ([True], [1]),
        ([1], [0.5]),
        ([1.0], [0]),
    ],
)
def test_nonbinary_or_unaligned_labels_rejected(predictions, labels):
    with pytest.raises(ValidationError):
        LabeledPredictionWindow(predictions=predictions, labels=labels, baseline_f1=0.9)


def test_confusion_counts_ignore_missing_labels():
    window = LabeledPredictionWindow(
        predictions=(1, 1, 0, 0, 1), labels=(1, 0, 1, 0, None), baseline_f1=0.9
    )
    assert binary_confusion_counts(window) == (1, 1, 1, 4, 2)


def test_degradation_uses_only_labeled_predictions(health_input):
    report = assess(health_input)
    assert report.assessment.decision == "RETRAIN_AND_EVALUATE"
    assert report.assessment.drift_detected
    assert report.performance.current == pytest.approx(2 / 3)
    assert report.performance.drop == pytest.approx(0.9 - 2 / 3)
    assert report.performance.ground_truth_coverage == 0.6
    assert report.performance.labeled_samples == 60
    assert report.performance.positive_labels == 20


def test_drift_without_ground_truth_requires_labels(health_input):
    body = health_input.model_dump(mode="json")
    body["performance"] = None
    report = assess(ModelHealthInput.model_validate(body))
    assert report.assessment.drift_detected
    assert report.assessment.decision == "COLLECT_LABELS"
    assert report.performance is None


@pytest.mark.parametrize(
    "predictions,labels",
    [
        ([0] * 100, [None] * 100),
        ([0] * 100, [0] * 100),
        ([1] * 100, [0] * 100),
        ([0] * 100, [1] * 4 + [0] * 96),
        ([0] * 100, [1] * 20 + [None] * 80),
    ],
)
def test_sparse_or_unrepresentative_labels_cannot_trigger_retraining(
    health_input, predictions, labels
):
    report = assess(with_performance(health_input, predictions, labels))
    assert report.assessment.decision == "COLLECT_LABELS"
    assert report.performance.sufficient_labels is False
    assert report.performance.degraded is False


def test_undefined_f1_is_reported_as_null(health_input):
    report = assess(with_performance(health_input, [0] * 100, [0] * 100))
    assert report.performance.current is None
    assert report.performance.drop is None


def test_stable_quality_with_drift_remains_healthy_and_flags_drift(health_input):
    labels = [1] * 20 + [0] * 80
    report = assess(with_performance(health_input, labels, labels))
    assert report.assessment.drift_detected
    assert report.assessment.decision == "HEALTHY"
    assert "monitored" in report.rationale_summary


def test_degradation_without_drift_still_recommends_evaluation(health_input):
    body = health_input.model_dump(mode="json")
    for item in body["distributions"]:
        item["current_counts"] = item["reference_counts"]
    report = assess(ModelHealthInput.model_validate(body))
    assert not report.assessment.drift_detected
    assert report.assessment.decision == "RETRAIN_AND_EVALUATE"


def test_f1_drop_threshold_is_inclusive(health_input):
    request = with_performance(health_input, [0] * 100, [1] * 20 + [0] * 80, baseline=0.05)
    assert assess(request).assessment.decision == "RETRAIN_AND_EVALUATE"
    smaller = with_performance(health_input, [0] * 100, [1] * 20 + [0] * 80, baseline=0.049)
    assert assess(smaller).assessment.decision == "HEALTHY"


def test_policy_overrides_quality_gate(health_input):
    policy = ModelHealthPolicy(min_ground_truth_coverage=0.9)
    assert assess(health_input, policy).assessment.decision == "COLLECT_LABELS"


@pytest.mark.parametrize(
    "policy",
    [
        {"psi_threshold": float("nan")},
        {"epsilon": 0},
        {"min_labeled_samples": 0},
        {"min_ground_truth_coverage": 1.1},
        {"max_f1_drop": float("inf")},
    ],
)
def test_invalid_policy_rejected(policy):
    with pytest.raises(ValidationError):
        ModelHealthPolicy(**policy)


def test_small_distributions_fail_explicitly(health_input):
    body = health_input.model_dump(mode="json")
    body["distributions"][0]["current_counts"] = [1, 1, 1, 1]
    with pytest.raises(InsufficientSamplesError, match="100"):
        assess(ModelHealthInput.model_validate(body))


@pytest.mark.parametrize(
    "field,value",
    [
        ("window_end", "2026-09-29T07:00:00Z"),
        ("window_start", "2026-09-29T08:00:00"),
        ("evidence_refs", ["duplicate", "duplicate"]),
    ],
)
def test_invalid_window_or_provenance_rejected(health_input, field, value):
    body = health_input.model_dump(mode="json")
    body[field] = value
    with pytest.raises(ValidationError):
        ModelHealthInput.model_validate(body)


def test_prediction_histogram_must_match_performance_window_size(health_input):
    body = health_input.model_dump(mode="json")
    body["distributions"][1]["current_counts"] = [200, 300, 500]
    with pytest.raises(ValidationError, match="match in size"):
        ModelHealthInput.model_validate(body)


def test_agent_returns_typed_result_without_fabricated_confidence(health_input):
    task = make_task(health_input)
    result = asyncio.run(MLOpsLifecycleAgent().handle(task))
    assert result.status is ResultStatus.SUCCEEDED
    payload = ModelHealthAssessed.model_validate(result.payload)
    assert payload.decision == "RETRAIN_AND_EVALUATE"
    assert result.task_id == task.task_id
    assert result.evidence_refs == health_input.evidence_refs
    assert result.confidence is None


@pytest.mark.parametrize("problem", ["missing_input", "unbounded_evidence", "small_window"])
def test_agent_invalid_input_returns_failure(health_input, problem):
    body = make_task(health_input).model_dump(mode="json")
    if problem == "missing_input":
        body["task_context"]["signals"] = {}
    elif problem == "unbounded_evidence":
        body["task_context"]["evidence_refs"] = []
    else:
        histograms = body["task_context"]["signals"]["model_health"]["distributions"]
        histograms[0]["current_counts"] = [1] * 4
    result = asyncio.run(MLOpsLifecycleAgent().handle(AgentTask.model_validate(body)))
    assert result.status is ResultStatus.FAILED
    assert result.payload == {}
    assert result.confidence is None


@pytest.mark.parametrize("problem", ["identity", "deadline"])
def test_agent_rejects_wrong_target_or_expired_deadline(health_input, problem):
    body = make_task(health_input).model_dump(mode="json")
    if problem == "identity":
        body["agent_id"] = AgentId.MONITORING.value
    else:
        body["deadline"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError):
        asyncio.run(MLOpsLifecycleAgent().handle(AgentTask.model_validate(body)))


def test_successful_mlops_result_requires_health_payload(health_input):
    task = make_task(health_input)
    with pytest.raises(ValidationError):
        AgentResult(
            result_id="result",
            task_id=task.task_id,
            incident_id=task.incident_id,
            agent_id=task.agent_id,
            status=ResultStatus.SUCCEEDED,
            rationale_summary="Invalid success without domain evidence",
        )
