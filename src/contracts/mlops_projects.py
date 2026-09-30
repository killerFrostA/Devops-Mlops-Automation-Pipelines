"""Project-neutral, versioned evidence and evaluation contracts for Agent 6."""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from src.contracts.base import Contract

Name = Annotated[str, Field(min_length=1, max_length=160)]
UnitScore = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
type TaskType = Literal["binary_classification", "regression", "text_generation", "custom"]


class ProjectManifest(Contract):
    """A project registers the exact model and evaluator policy it wants assessed."""

    project_id: Name
    model_id: Name
    model_version: Name
    task_type: TaskType
    evaluator_id: Name
    policy_version: Name
    min_samples: Annotated[int, Field(strict=True, ge=1)] = 30
    min_positive_samples: Annotated[int, Field(strict=True, ge=1)] = 5
    min_f1: UnitScore | None = None
    max_mae: Nonnegative | None = None
    min_text_score: UnitScore | None = None
    allow_external_llm: bool = False
    evaluator_config: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_policy(self) -> "ProjectManifest":
        expected = {
            "binary_classification": "min_f1",
            "regression": "max_mae",
            "text_generation": "min_text_score",
            "custom": None,
        }[self.task_type]
        present = {
            name
            for name in ("min_f1", "max_mae", "min_text_score")
            if getattr(self, name) is not None
        }
        required = {expected} if expected is not None else set()
        if present != required:
            raise ValueError(f"{self.task_type} requires only {expected or 'plugin configuration'}")
        if self.task_type != "custom" and self.evaluator_config:
            raise ValueError("evaluator_config is reserved for custom evaluators")
        if self.task_type == "text_generation" and self.min_samples > 100:
            raise ValueError("Text min_samples cannot exceed the evaluator batch limit of 100")
        if self.allow_external_llm and self.task_type != "text_generation":
            raise ValueError("External LLM access applies only to text-generation evaluations")
        return self


class BinaryEvaluationSample(Contract):
    task_type: Literal["binary_classification"] = "binary_classification"
    observation_id: Name
    observed_at: AwareDatetime
    expected: Literal[0, 1]
    predicted: Literal[0, 1]


class RegressionEvaluationSample(Contract):
    task_type: Literal["regression"] = "regression"
    observation_id: Name
    observed_at: AwareDatetime
    expected: FiniteNumber
    predicted: FiniteNumber


class CustomEvaluationSample(Contract):
    """Project-specific JSON payload; the registered plugin validates its contents."""

    task_type: Literal["custom"] = "custom"
    observation_id: Name
    observed_at: AwareDatetime
    payload: dict[str, JsonValue]


class TextEvaluationSample(Contract):
    task_type: Literal["text_generation"] = "text_generation"
    observation_id: Name
    observed_at: AwareDatetime
    prompt: Annotated[str, Field(min_length=1, max_length=8000)]
    output: Annotated[str, Field(min_length=1, max_length=8000)]
    reference: Annotated[str, Field(min_length=1, max_length=8000)]
    rubric: Annotated[str, Field(min_length=1, max_length=2000)]


type EvaluationSample = Annotated[
    BinaryEvaluationSample
    | RegressionEvaluationSample
    | TextEvaluationSample
    | CustomEvaluationSample,
    Field(discriminator="task_type"),
]


class EvaluationBatch(Contract):
    project_id: Name
    model_id: Name
    model_version: Name
    dataset_ref: Name
    evidence_ref: Name
    samples: Annotated[tuple[EvaluationSample, ...], Field(min_length=1, max_length=20_000)]

    @model_validator(mode="after")
    def unique_observations(self) -> "EvaluationBatch":
        ids = [sample.observation_id for sample in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("Evaluation observation IDs must be unique")
        return self


class TextJudgment(Contract):
    score: UnitScore
    rationale: Annotated[str, Field(min_length=1, max_length=1000)]


class ProjectEvaluationReport(Contract):
    project_id: Name
    model_id: Name
    model_version: Name
    task_type: TaskType
    evaluator_id: Name
    policy_version: Name
    dataset_ref: Name
    evidence_ref: Name
    sample_count: Annotated[int, Field(strict=True, ge=1)]
    decision: Literal["PASS", "FAIL", "INSUFFICIENT_EVIDENCE", "REVIEW_REQUIRED"]
    metrics: dict[str, FiniteNumber]
    rationale_summary: Annotated[str, Field(min_length=1)]
    requires_human_review: bool
    evaluator_model: str | None = None

    @model_validator(mode="after")
    def validate_review_gate(self) -> "ProjectEvaluationReport":
        if not self.requires_human_review:
            raise ValueError("Project evaluations must require review before promotion")
        if self.decision == "REVIEW_REQUIRED" and self.task_type not in (
            "text_generation",
            "custom",
        ):
            raise ValueError("REVIEW_REQUIRED is reserved for text or custom judgments")
        return self
