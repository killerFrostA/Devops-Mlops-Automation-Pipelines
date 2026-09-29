"""Inputs and diagnostic outputs for the first Agent 6 model-health milestone."""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from src.contracts.base import Contract
from src.contracts.payloads import ModelHealthAssessed

Name = Annotated[str, Field(min_length=1, max_length=128)]
Count = Annotated[int, Field(strict=True, ge=0, le=10**12)]
BinaryLabel = Annotated[int, Field(strict=True, ge=0, le=1)]
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class DistributionComparison(Contract):
    """The same fixed bins must be used for the reference and current window."""

    name: Name
    scope: Literal["feature", "prediction"]
    bins: Annotated[tuple[Name, ...], Field(min_length=2, max_length=100)]
    reference_counts: tuple[Count, ...]
    current_counts: tuple[Count, ...]

    @model_validator(mode="after")
    def validate_histograms(self) -> "DistributionComparison":
        if len(self.bins) != len(set(self.bins)):
            raise ValueError("Bin names must be unique")
        if len(self.bins) != len(self.reference_counts) or len(self.bins) != len(
            self.current_counts
        ):
            raise ValueError("Both count arrays must align with the same bins")
        if sum(self.reference_counts) == 0 or sum(self.current_counts) == 0:
            raise ValueError("Both distributions must contain observations")
        return self


class LabeledPredictionWindow(Contract):
    """Position i must identify the same prediction and ground-truth observation."""

    predictions: Annotated[tuple[BinaryLabel, ...], Field(min_length=1, max_length=10_000)]
    labels: tuple[BinaryLabel | None, ...]
    baseline_f1: Score

    @model_validator(mode="after")
    def validate_alignment(self) -> "LabeledPredictionWindow":
        if len(self.predictions) != len(self.labels):
            raise ValueError("Predictions and labels must align; use null for unavailable labels")
        return self


class ModelHealthInput(Contract):
    model_name: Name
    model_version: Name
    reference_id: Name
    window_start: AwareDatetime
    window_end: AwareDatetime
    distributions: Annotated[
        tuple[DistributionComparison, ...], Field(min_length=1, max_length=200)
    ]
    evidence_refs: Annotated[tuple[Name, ...], Field(min_length=1)]
    performance: LabeledPredictionWindow | None = None
    dataset_ref: Name | None = None

    @model_validator(mode="after")
    def validate_window(self) -> "ModelHealthInput":
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be later than window_start")
        keys = [(distribution.scope, distribution.name) for distribution in self.distributions]
        if len(keys) != len(set(keys)):
            raise ValueError("Distribution scope/name pairs must be unique")
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("Evidence references must be unique")
        if self.performance is not None:
            for distribution in self.distributions:
                if distribution.scope == "prediction" and sum(distribution.current_counts) != len(
                    self.performance.predictions
                ):
                    raise ValueError("Prediction histogram and labeled window must match in size")
        return self


class ModelHealthPolicy(Contract):
    """Initial tunable thresholds; require calibration on representative data."""

    version: Name = "agent6-health-v1"
    psi_threshold: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.2
    epsilon: Annotated[float, Field(ge=1e-12, le=0.01, allow_inf_nan=False)] = 1e-6
    min_distribution_samples: Annotated[int, Field(strict=True, ge=1)] = 100
    min_labeled_samples: Annotated[int, Field(strict=True, ge=1)] = 30
    min_positive_labels: Annotated[int, Field(strict=True, ge=1)] = 5
    min_ground_truth_coverage: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 0.5
    max_f1_drop: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 0.05


class DistributionHealth(Contract):
    name: Name
    scope: Literal["feature", "prediction"]
    method: Literal["PSI"] = "PSI"
    score: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    threshold: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    detected: bool
    reference_samples: Count
    current_samples: Count


class PerformanceHealth(Contract):
    metric: Literal["f1"] = "f1"
    baseline: Score
    current: Score | None
    drop: Score | None
    ground_truth_coverage: Score
    labeled_samples: Count
    positive_labels: Count
    true_positives: Count
    false_positives: Count
    false_negatives: Count
    sufficient_labels: bool
    degraded: bool


class ModelHealthReport(Contract):
    assessment: ModelHealthAssessed
    reference_id: Name
    window_start: AwareDatetime
    window_end: AwareDatetime
    distributions: tuple[DistributionHealth, ...]
    performance: PerformanceHealth | None
    policy: ModelHealthPolicy
    rationale_summary: str
    limitations: tuple[str, ...]
