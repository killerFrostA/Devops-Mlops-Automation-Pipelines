"""Pluggable, project-neutral evaluation with explicit evidence and review gates."""

from math import isfinite, sqrt
from typing import Literal, Protocol, cast

from src.contracts.mlops_projects import (
    BinaryEvaluationSample,
    EvaluationBatch,
    EvaluationSample,
    ProjectEvaluationReport,
    ProjectManifest,
    RegressionEvaluationSample,
    TaskType,
    TextEvaluationSample,
    TextJudgment,
)
from src.platform.errors import IntegrationNotConfiguredError


class TextJudge(Protocol):
    model_name: str

    async def judge(self, sample: TextEvaluationSample) -> TextJudgment: ...


class ProjectEvaluator(Protocol):
    evaluator_id: str
    task_type: TaskType

    async def evaluate(
        self, manifest: ProjectManifest, batch: EvaluationBatch
    ) -> ProjectEvaluationReport: ...


def _report(
    manifest: ProjectManifest,
    batch: EvaluationBatch,
    decision: Literal["PASS", "FAIL", "INSUFFICIENT_EVIDENCE", "REVIEW_REQUIRED"],
    metrics: dict[str, float],
    rationale: str,
    *,
    human_review: bool = True,
    evaluator_model: str | None = None,
) -> ProjectEvaluationReport:
    return ProjectEvaluationReport(
        project_id=manifest.project_id,
        model_id=manifest.model_id,
        model_version=manifest.model_version,
        task_type=manifest.task_type,
        evaluator_id=manifest.evaluator_id,
        policy_version=manifest.policy_version,
        dataset_ref=batch.dataset_ref,
        evidence_ref=batch.evidence_ref,
        sample_count=len(batch.samples),
        decision=decision,
        metrics=metrics,
        rationale_summary=rationale,
        requires_human_review=human_review,
        evaluator_model=evaluator_model,
    )


class BinaryClassificationEvaluator:
    evaluator_id = "binary-f1-v1"
    task_type: TaskType = "binary_classification"

    async def evaluate(
        self, manifest: ProjectManifest, batch: EvaluationBatch
    ) -> ProjectEvaluationReport:
        samples = _typed_samples(batch.samples, BinaryEvaluationSample)
        if len(samples) < manifest.min_samples:
            return _report(
                manifest,
                batch,
                "INSUFFICIENT_EVIDENCE",
                {},
                "Fewer labeled samples than the project policy requires",
            )
        positives = sum(row.expected for row in samples)
        if positives < manifest.min_positive_samples:
            return _report(
                manifest,
                batch,
                "INSUFFICIENT_EVIDENCE",
                {},
                "Too few positive reference labels for the project policy",
            )
        tp = sum(row.expected == 1 and row.predicted == 1 for row in samples)
        fp = sum(row.expected == 0 and row.predicted == 1 for row in samples)
        fn = positives - tp
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 0.0
        assert manifest.min_f1 is not None
        decision: Literal["PASS", "FAIL"] = "PASS" if f1 >= manifest.min_f1 else "FAIL"
        return _report(
            manifest,
            batch,
            decision,
            {"f1": f1, "positive_labels": float(positives)},
            f"F1 {f1:.3f} versus required {manifest.min_f1:.3f}; "
            "review cohort representativeness before promotion",
            human_review=True,
        )


class RegressionEvaluator:
    evaluator_id = "regression-mae-v1"
    task_type: TaskType = "regression"

    async def evaluate(
        self, manifest: ProjectManifest, batch: EvaluationBatch
    ) -> ProjectEvaluationReport:
        samples = _typed_samples(batch.samples, RegressionEvaluationSample)
        if len(samples) < manifest.min_samples:
            return _report(
                manifest,
                batch,
                "INSUFFICIENT_EVIDENCE",
                {},
                "Fewer labeled samples than the project policy requires",
            )
        errors = [row.predicted - row.expected for row in samples]
        mae = sum(abs(error) for error in errors) / len(errors)
        rmse = sqrt(sum(error * error for error in errors) / len(errors))
        if not isfinite(mae) or not isfinite(rmse):
            return _report(
                manifest,
                batch,
                "INSUFFICIENT_EVIDENCE",
                {},
                "Regression metric overflow; use a domain-appropriate scale",
            )
        assert manifest.max_mae is not None
        decision: Literal["PASS", "FAIL"] = "PASS" if mae <= manifest.max_mae else "FAIL"
        return _report(
            manifest,
            batch,
            decision,
            {"mae": mae, "rmse": rmse},
            f"MAE {mae:.3f} versus maximum {manifest.max_mae:.3f}; "
            "review cohort representativeness before promotion",
            human_review=True,
        )


class TextRubricEvaluator:
    evaluator_id = "text-rubric-llm-v1"
    task_type: TaskType = "text_generation"
    max_samples = 100

    def __init__(self, judge: TextJudge | None = None) -> None:
        self._judge = judge

    async def evaluate(
        self, manifest: ProjectManifest, batch: EvaluationBatch
    ) -> ProjectEvaluationReport:
        samples = _typed_samples(batch.samples, TextEvaluationSample)
        if len(samples) < manifest.min_samples:
            return _report(
                manifest,
                batch,
                "INSUFFICIENT_EVIDENCE",
                {},
                "Fewer reference-checked samples than the project policy requires",
                human_review=True,
            )
        if len(samples) > self.max_samples:
            return _report(
                manifest,
                batch,
                "INSUFFICIENT_EVIDENCE",
                {},
                f"Text batches are capped at {self.max_samples} samples for cost control",
                human_review=True,
            )
        if not manifest.allow_external_llm:
            raise IntegrationNotConfiguredError(
                "External LLM evaluation needs explicit project opt-in"
            )
        if self._judge is None:
            raise IntegrationNotConfiguredError("Text evaluation provider is not configured")
        judgments = [await self._judge.judge(sample) for sample in samples]
        mean = sum(judgment.score for judgment in judgments) / len(judgments)
        assert manifest.min_text_score is not None
        outcome = "above" if mean >= manifest.min_text_score else "below"
        return _report(
            manifest,
            batch,
            "REVIEW_REQUIRED",
            {"mean_rubric_score": mean},
            f"LLM rubric mean {mean:.3f} is {outcome} target "
            f"{manifest.min_text_score:.3f}; independent human review is required",
            human_review=True,
            evaluator_model=self._judge.model_name,
        )


def _typed_samples[
    SampleT: (BinaryEvaluationSample, RegressionEvaluationSample, TextEvaluationSample)
](samples: tuple[EvaluationSample, ...], sample_type: type[SampleT]) -> list[SampleT]:
    if any(not isinstance(sample, sample_type) for sample in samples):
        raise ValueError("Evaluation batch contains samples for a different task type")
    return cast(list[SampleT], list(samples))


class ProjectEvaluationService:
    """In-memory registration boundary; production persistence is supplied by orchestration."""

    def __init__(self, *, text_judge: TextJudge | None = None) -> None:
        self._projects: dict[tuple[str, str, str], ProjectManifest] = {}
        self._evaluators: dict[tuple[TaskType, str], ProjectEvaluator] = {}
        self.register_evaluator(BinaryClassificationEvaluator())
        self.register_evaluator(RegressionEvaluator())
        self.register_evaluator(TextRubricEvaluator(text_judge))

    def register_project(self, manifest: ProjectManifest) -> None:
        key = (manifest.project_id, manifest.model_id, manifest.model_version)
        previous = self._projects.get(key)
        if previous is not None and previous != manifest:
            raise ValueError("A registered model version cannot silently change policy")
        self._projects[key] = manifest

    def register_evaluator(self, evaluator: ProjectEvaluator) -> None:
        key = (evaluator.task_type, evaluator.evaluator_id)
        if key in self._evaluators:
            raise ValueError(f"Evaluator is already registered: {key}")
        self._evaluators[key] = evaluator

    async def evaluate(self, batch: EvaluationBatch) -> ProjectEvaluationReport:
        key = (batch.project_id, batch.model_id, batch.model_version)
        manifest = self._projects.get(key)
        if manifest is None:
            raise ValueError("Project and exact model version must be registered before evaluation")
        evaluator = self._evaluators.get((manifest.task_type, manifest.evaluator_id))
        if evaluator is None:
            raise IntegrationNotConfiguredError(
                f"No evaluator registered for {manifest.task_type}/{manifest.evaluator_id}"
            )
        if any(sample.task_type != manifest.task_type for sample in batch.samples):
            raise ValueError("Evaluation samples do not match the registered project task type")
        if len(batch.samples) < manifest.min_samples:
            return _report(
                manifest,
                batch,
                "INSUFFICIENT_EVIDENCE",
                {},
                "Fewer samples than the registered project policy requires",
            )
        report = await evaluator.evaluate(manifest, batch)
        expected = (
            manifest.project_id,
            manifest.model_id,
            manifest.model_version,
            manifest.task_type,
            manifest.evaluator_id,
            manifest.policy_version,
            batch.dataset_ref,
            batch.evidence_ref,
            len(batch.samples),
        )
        actual = (
            report.project_id,
            report.model_id,
            report.model_version,
            report.task_type,
            report.evaluator_id,
            report.policy_version,
            report.dataset_ref,
            report.evidence_ref,
            report.sample_count,
        )
        if actual != expected:
            raise ValueError("Evaluator report does not match registered project or evidence")
        return report
