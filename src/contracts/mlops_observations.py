"""Versioned prediction-window records for deterministic Agent 6 input construction."""

import math
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StrictFloat, StrictInt, StrictStr, model_validator

from src.contracts.base import Contract
from src.contracts.mlops import BinaryLabel, Count, Name, Score

FiniteEdge = Annotated[float, Field(allow_inf_nan=False)]
FeatureValue = StrictStr | StrictInt | StrictFloat | None


class NumericHistogramSpec(Contract):
    kind: Literal["numeric"] = "numeric"
    name: Name
    scope: Literal["feature", "prediction"]
    edges: Annotated[tuple[FiniteEdge, ...], Field(min_length=1, max_length=99)]
    bins: Annotated[tuple[Name, ...], Field(min_length=2, max_length=100)]
    missing_bin: Name | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "NumericHistogramSpec":
        if any(left >= right for left, right in zip(self.edges, self.edges[1:], strict=False)):
            raise ValueError("Numeric edges must increase strictly")
        if len(self.bins) != len(self.edges) + 1 + int(self.missing_bin is not None):
            raise ValueError("Numeric bins must match edges plus optional missing bin")
        if len(set(self.bins)) != len(self.bins):
            raise ValueError("Numeric bin names must be unique")
        if self.missing_bin is not None and self.bins[-1] != self.missing_bin:
            raise ValueError("The missing bin must be last")
        return self

    @property
    def bin_names(self) -> tuple[str, ...]:
        return self.bins


class CategoricalHistogramSpec(Contract):
    kind: Literal["categorical"] = "categorical"
    name: Name
    scope: Literal["feature"] = "feature"
    categories: Annotated[tuple[Name, ...], Field(min_length=1, max_length=98)]
    other_bin: Name
    missing_bin: Name | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "CategoricalHistogramSpec":
        names = (*self.categories, self.other_bin)
        if self.missing_bin is not None:
            names += (self.missing_bin,)
        if len(set(names)) != len(names) or len(names) > 100:
            raise ValueError("Categorical categories and bin names must be unique")
        return self

    @property
    def bin_names(self) -> tuple[str, ...]:
        names = (*self.categories, self.other_bin)
        return names + ((self.missing_bin,) if self.missing_bin is not None else ())


HistogramSpec = Annotated[
    NumericHistogramSpec | CategoricalHistogramSpec, Field(discriminator="kind")
]


class BaselineDistribution(Contract):
    spec: HistogramSpec
    reference_counts: tuple[Count, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> "BaselineDistribution":
        if len(self.reference_counts) != len(self.spec.bin_names):
            raise ValueError("Reference counts must match the baseline bin specification")
        if sum(self.reference_counts) == 0:
            raise ValueError("Reference distribution must contain observations")
        return self


class BaselineProfile(Contract):
    """Versioned reference values; a source adapter must establish provenance."""

    reference_id: Name
    model_name: Name
    model_version: Name
    baseline_f1: Score
    evidence_ref: Name
    distributions: Annotated[tuple[BaselineDistribution, ...], Field(min_length=1, max_length=200)]

    @model_validator(mode="after")
    def validate_distributions(self) -> "BaselineProfile":
        keys = [(item.spec.scope, item.spec.name) for item in self.distributions]
        if len(set(keys)) != len(keys):
            raise ValueError("Baseline distribution scope/name pairs must be unique")
        return self


class PredictionObservation(Contract):
    transaction_id: Name
    model_name: Name
    model_version: Name
    predicted_at: AwareDatetime
    features: dict[Name, FeatureValue]
    fraud_score: Score
    predicted_class: BinaryLabel

    @model_validator(mode="after")
    def validate_features(self) -> "PredictionObservation":
        for name, value in self.features.items():
            if isinstance(value, bool) or (
                isinstance(value, (float, int)) and not math.isfinite(value)
            ):
                raise ValueError(f"Feature {name} must be a finite number, string or null")
        return self


class LabelObservation(Contract):
    transaction_id: Name
    label: BinaryLabel
    confirmed_at: AwareDatetime


class WindowBuildRequest(Contract):
    baseline: BaselineProfile
    window_start: AwareDatetime
    window_end: AwareDatetime
    as_of: AwareDatetime
    predictions: Annotated[
        tuple[PredictionObservation, ...], Field(min_length=1, max_length=10_000)
    ]
    labels: Annotated[tuple[LabelObservation, ...], Field(max_length=10_000)] = ()
    predictions_evidence_ref: Name
    labels_evidence_ref: Name | None = None
    dataset_ref: Name | None = None

    @model_validator(mode="after")
    def validate_window(self) -> "WindowBuildRequest":
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be later than window_start")
        if self.as_of < self.window_end:
            raise ValueError("as_of must be at or after window_end")
        return self
