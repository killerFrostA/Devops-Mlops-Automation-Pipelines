"""Construct a validated health window from versioned records and fixed baseline bins."""

import math
from bisect import bisect_right

from src.contracts.mlops import DistributionComparison, LabeledPredictionWindow, ModelHealthInput
from src.contracts.mlops_observations import (
    CategoricalHistogramSpec,
    NumericHistogramSpec,
    PredictionObservation,
    WindowBuildRequest,
)


class WindowBuildError(ValueError):
    """Input records do not form one unambiguous model-health window."""


def build_model_health_input(request: WindowBuildRequest) -> ModelHealthInput:
    baseline = request.baseline
    predictions = tuple(
        record
        for record in request.predictions
        if request.window_start <= record.predicted_at < request.window_end
    )
    if not predictions:
        raise WindowBuildError("No predictions in the requested window")
    by_id: dict[str, PredictionObservation] = {}
    for record in predictions:
        if (record.model_name, record.model_version) != (
            baseline.model_name,
            baseline.model_version,
        ):
            raise WindowBuildError("Prediction model identity differs from the baseline")
        if record.observation_id in by_id:
            raise WindowBuildError(f"Duplicate prediction observation_id: {record.observation_id}")
        by_id[record.observation_id] = record

    labels: dict[str, int] = {}
    for label_record in request.labels:
        prediction = by_id.get(label_record.observation_id)
        if prediction is None or label_record.confirmed_at > request.as_of:
            continue
        if label_record.confirmed_at < prediction.predicted_at:
            raise WindowBuildError(
                f"Label predates its prediction for observation_id: {label_record.observation_id}"
            )
        if label_record.observation_id in labels:
            raise WindowBuildError(f"Duplicate confirmed label: {label_record.observation_id}")
        labels[label_record.observation_id] = label_record.label
    if labels and request.labels_evidence_ref is None:
        raise WindowBuildError("Confirmed labels require labels_evidence_ref")

    distributions = []
    for reference in baseline.distributions:
        spec = reference.spec
        counts = [0] * len(spec.bin_names)
        for record in predictions:
            if isinstance(spec, NumericHistogramSpec):
                value = (
                    record.positive_class_score
                    if spec.scope == "prediction"
                    else record.features.get(spec.name)
                )
                index = _numeric_bin(spec, value)
            else:
                index = _categorical_bin(spec, record.features.get(spec.name))
            counts[index] += 1
        distributions.append(
            DistributionComparison(
                name=spec.name,
                scope=spec.scope,
                bins=spec.bin_names,
                reference_counts=reference.reference_counts,
                current_counts=tuple(counts),
            )
        )

    evidence = [baseline.evidence_ref, request.predictions_evidence_ref]
    if labels:
        assert request.labels_evidence_ref is not None
        evidence.append(request.labels_evidence_ref)
    return ModelHealthInput(
        model_name=baseline.model_name,
        model_version=baseline.model_version,
        reference_id=baseline.reference_id,
        window_start=request.window_start,
        window_end=request.window_end,
        distributions=tuple(distributions),
        evidence_refs=tuple(evidence),
        performance=LabeledPredictionWindow(
            predictions=tuple(record.predicted_class for record in predictions),
            labels=tuple(labels.get(record.observation_id) for record in predictions),
            baseline_f1=baseline.baseline_f1,
        ),
        dataset_ref=request.dataset_ref,
    )


def _numeric_bin(spec: NumericHistogramSpec, value: object) -> int:
    if value is None:
        if spec.missing_bin is None:
            raise WindowBuildError(f"Missing value for numeric feature: {spec.name}")
        return len(spec.bins) - 1
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise WindowBuildError(f"Feature {spec.name} must be a finite number")
    return bisect_right(spec.edges, value)


def _categorical_bin(spec: CategoricalHistogramSpec, value: object) -> int:
    if value is None:
        if spec.missing_bin is None:
            raise WindowBuildError(f"Missing value for categorical feature: {spec.name}")
        return len(spec.bin_names) - 1
    if not isinstance(value, str):
        raise WindowBuildError(f"Feature {spec.name} must be a string")
    try:
        return spec.categories.index(value)
    except ValueError:
        return len(spec.categories)
