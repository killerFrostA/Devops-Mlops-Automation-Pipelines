"""Offline candidate-training contracts for the Agent 6 local demonstration."""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from src.contracts.base import Contract
from src.contracts.mlops import BinaryLabel, Count, ModelHealthReport, Name, Score
from src.contracts.mlops_observations import FeatureValue

FiniteNonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class TrainingObservation(Contract):
    observation_id: Name
    group_id: Name
    observed_at: AwareDatetime
    features: dict[Name, FeatureValue]
    label: BinaryLabel


class TrainingDataset(Contract):
    dataset_ref: Name
    model_name: Name
    observations: Annotated[tuple[TrainingObservation, ...], Field(min_length=1, max_length=20_000)]

    @model_validator(mode="after")
    def unique_observations(self) -> "TrainingDataset":
        ids = [row.observation_id for row in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("Training observation IDs must be unique")
        return self


class TrainingPolicy(Contract):
    version: Name = "agent6-training-demo-v1"
    pipeline_version: Name = "machine-logistic-v1"
    champion_features: Annotated[tuple[Name, ...], Field(min_length=1)]
    candidate_numeric_features: Annotated[tuple[Name, ...], Field(min_length=1)]
    candidate_categorical_features: tuple[Name, ...] = ()
    min_training_samples: Annotated[int, Field(strict=True, ge=1)] = 200
    min_validation_samples: Annotated[int, Field(strict=True, ge=1)] = 50
    min_test_samples: Annotated[int, Field(strict=True, ge=1)] = 50
    min_positive_per_split: Annotated[int, Field(strict=True, ge=1)] = 5
    min_f1_gain: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.02
    max_p95_latency_ms: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 200.0
    require_recall_nonregression: bool = True
    random_state: int = 42

    @model_validator(mode="after")
    def validate_features(self) -> "TrainingPolicy":
        candidate = (
            *self.candidate_numeric_features,
            *self.candidate_categorical_features,
        )
        if len(candidate) != len(set(candidate)):
            raise ValueError("Candidate feature names must be unique")
        if len(self.champion_features) != len(set(self.champion_features)):
            raise ValueError("Champion feature names must be unique")
        if not set(self.champion_features).issubset(self.candidate_numeric_features):
            raise ValueError("Champion features must be a subset of candidate numeric features")
        return self


class ClassificationMetrics(Contract):
    f1: Score
    precision: Score
    recall: Score
    accuracy: Score
    p95_latency_ms: FiniteNonnegative
    true_positives: Count
    false_positives: Count
    false_negatives: Count
    true_negatives: Count


class CandidateEvaluation(Contract):
    model_name: Name
    pipeline_version: Name
    dataset_ref: Name
    dataset_sha256: Name
    run_id: Name
    candidate_model_uri: Name
    train_samples: Count
    validation_samples: Count
    test_samples: Count
    champion: ClassificationMetrics
    candidate: ClassificationMetrics
    candidate_validation: ClassificationMetrics
    policy: TrainingPolicy


class ModelLifecycleReport(Contract):
    health: ModelHealthReport
    evaluation: CandidateEvaluation | None
    decision: Literal["SKIP_TRAINING", "KEEP_CHAMPION", "REQUEST_PROMOTION_REVIEW"]
    registered_model_version: str | None = None
    rationale_summary: str
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_decision(self) -> "ModelLifecycleReport":
        if self.decision == "SKIP_TRAINING" and (
            self.evaluation is not None or self.registered_model_version is not None
        ):
            raise ValueError("Skipped training cannot include an evaluation or registered version")
        if self.decision == "KEEP_CHAMPION" and (
            self.evaluation is None or self.registered_model_version is not None
        ):
            raise ValueError("Keeping champion requires evaluation without registration")
        if self.decision == "REQUEST_PROMOTION_REVIEW" and (
            self.evaluation is None or self.registered_model_version is None
        ):
            raise ValueError("Promotion review requires evaluation and a registered version")
        return self
