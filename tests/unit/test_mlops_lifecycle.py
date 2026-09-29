"""Agent 6 lifecycle gates and split boundaries."""

import asyncio
from pathlib import Path

import pytest

from src.agents.mlops_lifecycle.demo_data import synthetic_machine_training_dataset
from src.agents.mlops_lifecycle.health import RuleBasedModelHealthAssessor
from src.agents.mlops_lifecycle.local_files import (
    FileBaselineRepository,
    FileLabelRepository,
    FilePredictionRepository,
    LocalSourceConfig,
)
from src.agents.mlops_lifecycle.service import MLOpsLifecycleAgent
from src.agents.mlops_lifecycle.training import split_dataset
from src.contracts.mlops import ModelHealthPolicy
from src.contracts.mlops_training import (
    CandidateEvaluation,
    ClassificationMetrics,
    TrainingDataset,
    TrainingPolicy,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "contracts/examples/agent6/local-sources/machine-sources.json"


def _policy() -> TrainingPolicy:
    return TrainingPolicy.model_validate_json(
        (ROOT / "configs/agent6-training-demo.json").read_text(encoding="utf-8")
    )


def _agent(
    training: object, registry: object, assessor: RuleBasedModelHealthAssessor | None = None
) -> tuple[MLOpsLifecycleAgent, LocalSourceConfig]:
    source = LocalSourceConfig.model_validate_json(SOURCE.read_text(encoding="utf-8"))
    return (
        MLOpsLifecycleAgent(
            assessor=assessor,
            baselines=FileBaselineRepository(SOURCE.parent / source.baseline_file),
            predictions=FilePredictionRepository(SOURCE.parent / source.predictions_file),
            labels=FileLabelRepository(SOURCE.parent / source.labels_file),
            training=training,  # type: ignore[arg-type]
            registry=registry,  # type: ignore[arg-type]
        ),
        source,
    )


def _metrics(f1: float, recall: float = 0.5, latency: float = 2.0) -> ClassificationMetrics:
    return ClassificationMetrics(
        f1=f1,
        precision=f1,
        recall=recall,
        accuracy=f1,
        p95_latency_ms=latency,
        true_positives=10,
        false_positives=10,
        false_negatives=10,
        true_negatives=10,
    )


class StubTraining:
    def __init__(self, candidate: ClassificationMetrics) -> None:
        self.candidate = candidate
        self.calls = 0

    async def train_and_evaluate(
        self, dataset: TrainingDataset, policy: TrainingPolicy
    ) -> CandidateEvaluation:
        self.calls += 1
        return CandidateEvaluation(
            model_name=dataset.model_name,
            pipeline_version=policy.pipeline_version,
            dataset_ref=dataset.dataset_ref,
            dataset_sha256="a" * 64,
            run_id="run1",
            candidate_model_uri="models:/candidate",
            train_samples=360,
            validation_samples=120,
            test_samples=120,
            champion=_metrics(0.4),
            candidate=self.candidate,
            candidate_validation=self.candidate,
            policy=policy,
        )


class StubRegistry:
    def __init__(self) -> None:
        self.calls = 0

    async def register_candidate(self, evaluation: CandidateEvaluation) -> str:
        self.calls += 1
        assert evaluation.candidate_model_uri == "models:/candidate"
        assert evaluation.model_name == "machine-failure"
        return "1"


def test_synthetic_data_and_splits_are_reproducible_and_disjoint() -> None:
    dataset = synthetic_machine_training_dataset()
    assert dataset == synthetic_machine_training_dataset()
    train, validation, test = split_dataset(dataset, _policy())
    assert (len(train), len(validation), len(test)) == (360, 120, 120)
    group_sets = [{row.group_id for row in rows} for rows in (train, validation, test)]
    assert not (
        group_sets[0] & group_sets[1]
        | group_sets[1] & group_sets[2]
        | group_sets[0] & group_sets[2]
    )
    assert max(row.observed_at for row in train) < min(row.observed_at for row in validation)
    assert max(row.observed_at for row in validation) < min(row.observed_at for row in test)


def test_split_rejects_insufficient_class_examples() -> None:
    dataset = synthetic_machine_training_dataset()
    policy = _policy().model_copy(update={"min_positive_per_split": 120})
    with pytest.raises(ValueError, match="examples per class"):
        split_dataset(dataset, policy)


def test_qualified_candidate_requests_review() -> None:
    training = StubTraining(_metrics(0.8, recall=0.8))
    registry = StubRegistry()
    agent, source = _agent(training, registry)
    report = asyncio.run(
        agent.assess_and_train(source.query, synthetic_machine_training_dataset(), _policy())
    )
    assert report.health.assessment.decision == "RETRAIN_AND_EVALUATE"
    assert report.decision == "REQUEST_PROMOTION_REVIEW"
    assert report.registered_model_version == "1"
    assert registry.calls == 1


def test_weak_candidate_keeps_champion_without_registration() -> None:
    training = StubTraining(_metrics(0.41, recall=0.4))
    registry = StubRegistry()
    agent, source = _agent(training, registry)
    report = asyncio.run(
        agent.assess_and_train(source.query, synthetic_machine_training_dataset(), _policy())
    )
    assert report.decision == "KEEP_CHAMPION"
    assert report.registered_model_version is None
    assert registry.calls == 0


def test_training_dataset_must_predate_window() -> None:
    training = StubTraining(_metrics(0.8, recall=0.8))
    registry = StubRegistry()
    agent, source = _agent(training, registry)
    dataset = synthetic_machine_training_dataset()
    late = dataset.observations[0].model_copy(update={"observed_at": source.query.window_start})
    dataset = dataset.model_copy(update={"observations": (late, *dataset.observations[1:])})
    with pytest.raises(ValueError, match="predate"):
        asyncio.run(agent.assess_and_train(source.query, dataset, _policy()))
    assert training.calls == 0


def test_healthy_window_skips_training() -> None:
    training = StubTraining(_metrics(0.8, recall=0.8))
    registry = StubRegistry()
    assessor = RuleBasedModelHealthAssessor(ModelHealthPolicy(max_f1_drop=0.9))
    agent, source = _agent(training, registry, assessor)
    report = asyncio.run(
        agent.assess_and_train(source.query, synthetic_machine_training_dataset(), _policy())
    )
    assert report.health.assessment.decision == "HEALTHY"
    assert report.decision == "SKIP_TRAINING"
    assert report.evaluation is None
    assert training.calls == 0
    assert registry.calls == 0
