"""Reproducible, local candidate training with a held-out chronological test."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from statistics import quantiles
from typing import Any

from src.contracts.mlops_training import (
    CandidateEvaluation,
    ClassificationMetrics,
    TrainingDataset,
    TrainingObservation,
    TrainingPolicy,
)


def split_dataset(
    dataset: TrainingDataset, policy: TrainingPolicy
) -> tuple[
    tuple[TrainingObservation, ...],
    tuple[TrainingObservation, ...],
    tuple[TrainingObservation, ...],
]:
    """Sort by time and put whole machine groups in only one split."""
    ordered = sorted(dataset.observations, key=lambda row: (row.observed_at, row.observation_id))
    groups: dict[str, list[TrainingObservation]] = {}
    for row in ordered:
        groups.setdefault(row.group_id, []).append(row)
    group_rows = sorted(
        groups.values(), key=lambda rows: (min(row.observed_at for row in rows), rows[0].group_id)
    )
    n_groups = len(group_rows)
    if n_groups < 3:
        raise ValueError("At least three distinct machine groups are required")
    boundaries = (int(n_groups * 0.6), int(n_groups * 0.8))
    chunks = (
        group_rows[: boundaries[0]],
        group_rows[boundaries[0] : boundaries[1]],
        group_rows[boundaries[1] :],
    )
    splits = tuple(tuple(row for group in chunk for row in group) for chunk in chunks)
    minimums = (policy.min_training_samples, policy.min_validation_samples, policy.min_test_samples)
    for name, rows, minimum in zip(
        ("training", "validation", "test"), splits, minimums, strict=True
    ):
        if len(rows) < minimum:
            raise ValueError(f"{name} split needs at least {minimum} observations")
        positives = sum(row.label for row in rows)
        if (
            positives < policy.min_positive_per_split
            or len(rows) - positives < policy.min_positive_per_split
        ):
            raise ValueError(
                f"{name} split needs at least {policy.min_positive_per_split} examples per class"
            )
    if max(row.observed_at for row in splits[0]) >= min(row.observed_at for row in splits[1]):
        raise ValueError("Machine groups overlap training and validation time periods")
    if max(row.observed_at for row in splits[1]) >= min(row.observed_at for row in splits[2]):
        raise ValueError("Machine groups overlap validation and test time periods")
    return splits[0], splits[1], splits[2]


def _feature_rows(
    rows: tuple[TrainingObservation, ...], features: tuple[str, ...]
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in rows:
        missing = set(features) - row.features.keys()
        if missing:
            raise ValueError(f"Observation {row.observation_id} lacks features: {sorted(missing)}")
        result.append({name: row.features[name] for name in features})
    return result


def _metrics(
    model: Any, rows: tuple[TrainingObservation, ...], features: tuple[str, ...]
) -> ClassificationMetrics:
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    x = _feature_rows(rows, features)
    y = [row.label for row in rows]
    durations = []
    predictions = []
    for item in x:
        started = time.perf_counter_ns()
        predictions.append(int(model.predict(pd.DataFrame([item]))[0]))
        durations.append((time.perf_counter_ns() - started) / 1_000_000)
    tn, fp, fn, tp = (
        int(value) for value in confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    )
    # Local p95 is only an approximate latency signal.
    p95 = quantiles(durations, n=100)[94]
    return ClassificationMetrics(
        f1=float(f1_score(y, predictions, zero_division=0)),
        precision=float(precision_score(y, predictions, zero_division=0)),
        recall=float(recall_score(y, predictions, zero_division=0)),
        accuracy=float(accuracy_score(y, predictions)),
        p95_latency_ms=p95,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
    )


class VibrationThresholdChampion:
    """Synthetic legacy rule: alert when vibration is at least 9 mm/s."""

    def predict(self, rows: Any) -> list[int]:
        return [int(value >= 9.0) for value in rows["vibration_mm_s"]]


def _pipeline(numeric: tuple[str, ...], categorical: tuple[str, ...], seed: int) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformers: list[tuple[str, Any, list[str]]] = [
        (
            "numeric",
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            list(numeric),
        )
    ]
    if categorical:
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(categorical))
        )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers=transformers)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )


class SklearnTrainingPipeline:
    """Train only on historical examples; MLflow stores evaluation evidence and model."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()

    async def train_and_evaluate(
        self, dataset: TrainingDataset, policy: TrainingPolicy
    ) -> CandidateEvaluation:
        import mlflow
        import mlflow.sklearn
        import pandas as pd

        self.output_dir.mkdir(parents=True, exist_ok=True)
        train, validation, test = split_dataset(dataset, policy)
        features = (*policy.candidate_numeric_features, *policy.candidate_categorical_features)
        digest = hashlib.sha256(
            dataset.model_dump_json(exclude_none=True).encode("utf-8")
        ).hexdigest()
        champion = VibrationThresholdChampion()
        candidate = _pipeline(
            policy.candidate_numeric_features,
            policy.candidate_categorical_features,
            policy.random_state,
        )
        candidate.fit(pd.DataFrame(_feature_rows(train, features)), [row.label for row in train])
        champion_test = _metrics(champion, test, policy.champion_features)
        candidate_validation = _metrics(candidate, validation, features)
        candidate_test = _metrics(candidate, test, features)

        # SQLite is a local tracking and registry backend. All state stays under output_dir.
        uri = "sqlite:///" + (self.output_dir / "mlflow.db").as_posix()
        mlflow.set_tracking_uri(uri)
        experiment_name = "agent6-machine-demo-local"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else mlflow.create_experiment(
                experiment_name,
                artifact_location=(self.output_dir / "mlruns").as_uri(),
            )
        )
        with mlflow.start_run(
            experiment_id=experiment_id, run_name=f"{dataset.model_name}-{digest[:10]}"
        ) as run:
            mlflow.log_params(
                {
                    "dataset_ref": dataset.dataset_ref,
                    "dataset_sha256": digest,
                    "pipeline_version": policy.pipeline_version,
                    "policy_version": policy.version,
                    "train_samples": len(train),
                    "validation_samples": len(validation),
                    "test_samples": len(test),
                    "champion_source": "synthetic-vibration-threshold-9mm-s-proxy",
                }
            )
            mlflow.log_metrics(
                {
                    "champion_test_f1": champion_test.f1,
                    "champion_test_recall": champion_test.recall,
                    "candidate_validation_f1": candidate_validation.f1,
                    "candidate_test_f1": candidate_test.f1,
                    "candidate_test_recall": candidate_test.recall,
                    "candidate_test_p95_latency_ms": candidate_test.p95_latency_ms,
                }
            )
            mlflow.log_dict(policy.model_dump(mode="json"), "training_policy.json")
            mlflow.log_dict(
                {
                    "train_ids": [row.observation_id for row in train],
                    "validation_ids": [row.observation_id for row in validation],
                    "test_ids": [row.observation_id for row in test],
                },
                "split_manifest.json",
            )
            model_info = mlflow.sklearn.log_model(
                sk_model=candidate,
                name="candidate",
                skops_trusted_types=["numpy.dtype"],
                input_example=pd.DataFrame(_feature_rows(train[:2], features)),
            )
            return CandidateEvaluation(
                model_name=dataset.model_name,
                pipeline_version=policy.pipeline_version,
                dataset_ref=dataset.dataset_ref,
                dataset_sha256=digest,
                run_id=run.info.run_id,
                candidate_model_uri=model_info.model_uri,
                train_samples=len(train),
                validation_samples=len(validation),
                test_samples=len(test),
                champion=champion_test,
                candidate=candidate_test,
                candidate_validation=candidate_validation,
                policy=policy,
            )
