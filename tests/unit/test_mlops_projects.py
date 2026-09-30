"""Project-neutral Agent 6 evaluation and bounded task handling."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from src.agents.mlops_lifecycle.evaluation.engine import ProjectEvaluationService
from src.agents.mlops_lifecycle.service import MLOpsLifecycleAgent
from src.contracts.enums import AgentId, Environment, ResultStatus
from src.contracts.mlops_projects import (
    EvaluationBatch,
    ProjectEvaluationReport,
    ProjectManifest,
    TextEvaluationSample,
    TextJudgment,
)
from src.contracts.tasks import AgentTask, TaskContext
from src.platform.errors import IntegrationNotConfiguredError

NOW = datetime(2026, 9, 30, tzinfo=UTC)


def manifest(task: str, **overrides: object) -> ProjectManifest:
    policies = {
        "binary_classification": {"evaluator_id": "binary-f1-v1", "min_f1": 0.8},
        "regression": {"evaluator_id": "regression-mae-v1", "max_mae": 0.25},
        "text_generation": {
            "evaluator_id": "text-rubric-llm-v1",
            "min_text_score": 0.75,
            "allow_external_llm": True,
        },
    }
    return ProjectManifest.model_validate(
        {
            "project_id": f"project-{task}",
            "model_id": "model-a",
            "model_version": "v1",
            "task_type": task,
            "policy_version": "policy-v1",
            "min_samples": 2,
            "min_positive_samples": 1,
            **policies[task],
            **overrides,
        }
    )


def batch(task: str, pairs: list[tuple[object, object]]) -> EvaluationBatch:
    records = []
    for index, (expected, predicted) in enumerate(pairs):
        record = {
            "task_type": task,
            "observation_id": f"item-{index}",
            "observed_at": NOW.isoformat(),
        }
        if task == "text_generation":
            record.update(
                prompt=f"Question {index}",
                output=str(predicted),
                reference=str(expected),
                rubric="Evaluate factual correctness against the reference",
            )
        else:
            record.update(expected=expected, predicted=predicted)
        records.append(record)
    return EvaluationBatch.model_validate(
        {
            "project_id": f"project-{task}",
            "model_id": "model-a",
            "model_version": "v1",
            "dataset_ref": "heldout-2026-09",
            "evidence_ref": "sha256:1234",
            "samples": records,
        }
    )


def test_two_unrelated_projects_use_independent_metrics_and_policies() -> None:
    service = ProjectEvaluationService()
    service.register_project(manifest("binary_classification"))
    service.register_project(manifest("regression"))
    classification = asyncio.run(
        service.evaluate(batch("binary_classification", [(1, 1), (0, 0), (1, 1)]))
    )
    regression = asyncio.run(service.evaluate(batch("regression", [(10.0, 10.8), (5.0, 5.0)])))
    assert classification.decision == "PASS"
    assert classification.metrics["f1"] == 1.0
    assert regression.decision == "FAIL"
    assert regression.metrics["mae"] == pytest.approx(0.4)
    assert classification.requires_human_review
    assert regression.requires_human_review


def test_insufficient_or_mismatched_evidence_fails_closed() -> None:
    service = ProjectEvaluationService()
    service.register_project(manifest("binary_classification"))
    short = asyncio.run(service.evaluate(batch("binary_classification", [(1, 1)])))
    no_positives = asyncio.run(service.evaluate(batch("binary_classification", [(0, 0), (0, 0)])))
    assert short.decision == "INSUFFICIENT_EVIDENCE"
    assert no_positives.decision == "INSUFFICIENT_EVIDENCE"
    with pytest.raises(ValueError, match="task type"):
        asyncio.run(
            service.evaluate(
                batch("binary_classification", [(1, 1)]).model_copy(
                    update={"samples": batch("regression", [(1.0, 1.0)]).samples}
                )
            )
        )
    with pytest.raises(ValueError, match="registered"):
        service.register_project(manifest("binary_classification", min_f1=0.2))


def test_contracts_reject_mixed_ids_invalid_policy_and_nonfinite_numbers() -> None:
    with pytest.raises(ValidationError):
        manifest("regression", min_f1=0.9)
    with pytest.raises(ValidationError):
        batch("regression", [(1.0, float("nan"))])
    duplicated = batch("regression", [(1.0, 1.0), (2.0, 2.0)]).model_dump(mode="json")
    duplicated["samples"][1]["observation_id"] = "item-0"
    with pytest.raises(ValidationError):
        EvaluationBatch.model_validate(duplicated)


class StubJudge:
    model_name = "stub:judge-v1"

    def __init__(self) -> None:
        self.calls = 0

    async def judge(self, sample: TextEvaluationSample) -> TextJudgment:
        self.calls += 1
        return TextJudgment(score=0.9, rationale=f"Reference matched for {sample.observation_id}")


def test_text_judgment_needs_opt_in_and_human_review() -> None:
    judge = StubJudge()
    service = ProjectEvaluationService(text_judge=judge)
    service.register_project(manifest("text_generation", allow_external_llm=False))
    text_batch = batch("text_generation", [("Paris", "Paris"), ("Tunis", "Tunis")])
    with pytest.raises(IntegrationNotConfiguredError, match="opt-in"):
        asyncio.run(service.evaluate(text_batch))
    assert judge.calls == 0
    service = ProjectEvaluationService(text_judge=judge)
    service.register_project(manifest("text_generation"))
    report = asyncio.run(service.evaluate(text_batch))
    assert report.decision == "REVIEW_REQUIRED"
    assert report.metrics["mean_rubric_score"] == pytest.approx(0.9)
    assert report.evaluator_model == "stub:judge-v1"
    assert report.requires_human_review
    assert "Paris" not in report.model_dump_json()


def test_orchestrator_task_requires_pre_registered_project_and_scoped_evidence() -> None:
    agent = MLOpsLifecycleAgent()
    agent.register_project(manifest("regression"))
    payload = batch("regression", [(1.0, 1.0), (2.0, 2.0)]).model_dump(mode="json")

    def make_task(evidence: tuple[str, ...]) -> AgentTask:
        return AgentTask(
            task_id="task-1",
            incident_id="incident-1",
            agent_id=AgentId.MLOPS_LIFECYCLE,
            objective="Assess the registered model version",
            task_context=TaskContext(
                service_id="service-1",
                namespace="test",
                environment=Environment.TEST,
                signals={"project_evaluation": payload},
                evidence_refs=evidence,
            ),
            deadline=datetime.now(UTC) + timedelta(minutes=5),
        )

    failed = asyncio.run(agent.handle(make_task(())))
    assert failed.status is ResultStatus.FAILED
    passed = asyncio.run(agent.handle(make_task(("sha256:1234",))))
    assert passed.status is ResultStatus.SUCCEEDED
    assert passed.payload["decision"] == "PASS"
    assert passed.evidence_refs == ("sha256:1234",)


class CustomEvaluator:
    evaluator_id = "document-check-v1"
    task_type = "custom"

    async def evaluate(
        self, manifest: ProjectManifest, batch: EvaluationBatch
    ) -> ProjectEvaluationReport:
        accepted = []
        for sample in batch.samples:
            assert sample.task_type == "custom"
            value = sample.payload.get("accepted")
            if not isinstance(value, bool):
                raise ValueError("custom payload requires accepted boolean")
            accepted.append(value)
        rate = sum(accepted) / len(accepted)
        target = manifest.evaluator_config["required_success_rate"]
        assert isinstance(target, float)
        return ProjectEvaluationReport(
            project_id=manifest.project_id,
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            task_type="custom",
            evaluator_id=manifest.evaluator_id,
            policy_version=manifest.policy_version,
            dataset_ref=batch.dataset_ref,
            evidence_ref=batch.evidence_ref,
            sample_count=len(batch.samples),
            decision="PASS" if rate >= target else "FAIL",
            metrics={"success_rate": rate},
            rationale_summary="Custom document check evaluated",
            requires_human_review=True,
        )


def test_custom_project_plugin_uses_same_registration_and_evidence_boundary() -> None:
    service = ProjectEvaluationService()
    service.register_evaluator(CustomEvaluator())
    service.register_project(
        ProjectManifest(
            project_id="documents",
            model_id="extractor",
            model_version="v3",
            task_type="custom",
            evaluator_id="document-check-v1",
            policy_version="doc-v1",
            min_samples=2,
            evaluator_config={"required_success_rate": 1.0},
        )
    )
    custom_batch = EvaluationBatch.model_validate(
        {
            "project_id": "documents",
            "model_id": "extractor",
            "model_version": "v3",
            "dataset_ref": "documents-heldout",
            "evidence_ref": "sha256:abcd",
            "samples": [
                {
                    "task_type": "custom",
                    "observation_id": "a",
                    "observed_at": NOW.isoformat(),
                    "payload": {"accepted": True},
                },
                {
                    "task_type": "custom",
                    "observation_id": "b",
                    "observed_at": NOW.isoformat(),
                    "payload": {"accepted": False},
                },
            ],
        }
    )
    report = asyncio.run(service.evaluate(custom_batch))
    assert report.decision == "FAIL"
    assert report.metrics["success_rate"] == 0.5
    with pytest.raises(ValueError, match="registered"):
        asyncio.run(ProjectEvaluationService().evaluate(custom_batch))


def test_custom_plugin_cannot_relabel_evidence_or_bypass_minimum_samples() -> None:
    class TamperingEvaluator(CustomEvaluator):
        calls = 0

        async def evaluate(
            self, manifest: ProjectManifest, batch: EvaluationBatch
        ) -> ProjectEvaluationReport:
            self.calls += 1
            report = await super().evaluate(manifest, batch)
            return report.model_copy(update={"evidence_ref": "another-dataset"})

    evaluator = TamperingEvaluator()
    service = ProjectEvaluationService()
    service.register_evaluator(evaluator)
    service.register_project(
        ProjectManifest(
            project_id="documents",
            model_id="extractor",
            model_version="v3",
            task_type="custom",
            evaluator_id="document-check-v1",
            policy_version="doc-v1",
            min_samples=2,
            evaluator_config={"required_success_rate": 1.0},
        )
    )
    small = EvaluationBatch.model_validate(
        {
            "project_id": "documents",
            "model_id": "extractor",
            "model_version": "v3",
            "dataset_ref": "documents-heldout",
            "evidence_ref": "sha256:abcd",
            "samples": [
                {
                    "task_type": "custom",
                    "observation_id": "a",
                    "observed_at": NOW.isoformat(),
                    "payload": {"accepted": True},
                }
            ],
        }
    )
    report = asyncio.run(service.evaluate(small))
    assert report.decision == "INSUFFICIENT_EVIDENCE"
    assert evaluator.calls == 0
    large = small.model_copy(
        update={
            "samples": (
                *small.samples,
                small.samples[0].model_copy(update={"observation_id": "b"}),
            )
        }
    )
    with pytest.raises(ValueError, match="registered project or evidence"):
        asyncio.run(service.evaluate(large))
