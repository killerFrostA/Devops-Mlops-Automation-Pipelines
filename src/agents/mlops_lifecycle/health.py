"""Local model-health analysis; no retraining, deployment or storage side effects."""

from typing import Literal
from uuid import uuid4

from src.agents.mlops_lifecycle.metrics import binary_confusion_counts, population_stability_index
from src.contracts.mlops import (
    DistributionHealth,
    ModelHealthInput,
    ModelHealthPolicy,
    ModelHealthReport,
    PerformanceHealth,
)
from src.contracts.payloads import ModelHealthAssessed


class InsufficientSamplesError(ValueError):
    """A distribution window is too small for this configured assessment policy."""


class RuleBasedModelHealthAssessor:
    def __init__(self, policy: ModelHealthPolicy | None = None) -> None:
        self.policy = policy or ModelHealthPolicy()

    async def assess(self, request: ModelHealthInput) -> ModelHealthReport:
        distributions = []
        for comparison in request.distributions:
            reference_size = sum(comparison.reference_counts)
            current_size = sum(comparison.current_counts)
            if min(reference_size, current_size) < self.policy.min_distribution_samples:
                raise InsufficientSamplesError(
                    f"{comparison.name}: both distributions need at least "
                    f"{self.policy.min_distribution_samples} observations"
                )
            score = population_stability_index(comparison, self.policy.epsilon)
            distributions.append(
                DistributionHealth(
                    name=comparison.name,
                    scope=comparison.scope,
                    score=score,
                    threshold=self.policy.psi_threshold,
                    detected=score >= self.policy.psi_threshold,
                    reference_samples=reference_size,
                    current_samples=current_size,
                )
            )

        performance = self._performance(request)
        drift_detected = any(item.detected for item in distributions)
        decision: Literal["HEALTHY", "COLLECT_LABELS", "RETRAIN_AND_EVALUATE"]
        if performance is None or not performance.sufficient_labels:
            decision = "COLLECT_LABELS"
            rationale = (
                "Ground truth is absent, sparse or lacks enough positive examples; "
                "collect representative labels before judging model quality"
            )
        elif performance.degraded:
            decision = "RETRAIN_AND_EVALUATE"
            rationale = "Labeled F1 degradation meets or exceeds the policy threshold"
        else:
            decision = "HEALTHY"
            rationale = "Labeled F1 remains within the configured degradation tolerance"
            if drift_detected:
                rationale += "; distribution drift is present and should continue to be monitored"

        assessment = ModelHealthAssessed(
            assessment_id=str(uuid4()),
            model_name=request.model_name,
            model_version=request.model_version,
            drift_detected=drift_detected,
            drift_score=max(item.score for item in distributions),
            decision=decision,
            evidence_refs=request.evidence_refs,
            dataset_ref=request.dataset_ref,
        )
        return ModelHealthReport(
            assessment=assessment,
            reference_id=request.reference_id,
            window_start=request.window_start,
            window_end=request.window_end,
            distributions=tuple(distributions),
            performance=performance,
            policy=self.policy,
            rationale_summary=rationale,
            limitations=(
                "PSI measures distribution shift and does not prove accuracy degradation.",
                "F1 depends on representative labels aligned with the evaluated predictions.",
                "Policy thresholds require calibration; no decision confidence is fabricated.",
                "This report recommends a next step; training and promotion remain unimplemented.",
            ),
        )

    def _performance(self, request: ModelHealthInput) -> PerformanceHealth | None:
        window = request.performance
        if window is None:
            return None
        tp, fp, fn, labeled, positives = binary_confusion_counts(window)
        denominator = 2 * tp + fp + fn
        current = 2 * tp / denominator if denominator else None
        drop = max(0.0, window.baseline_f1 - current) if current is not None else None
        coverage = labeled / len(window.predictions)
        sufficient = (
            current is not None
            and labeled >= self.policy.min_labeled_samples
            and positives >= self.policy.min_positive_labels
            and coverage >= self.policy.min_ground_truth_coverage
        )
        return PerformanceHealth(
            baseline=window.baseline_f1,
            current=current,
            drop=drop,
            ground_truth_coverage=coverage,
            labeled_samples=labeled,
            positive_labels=positives,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            sufficient_labels=sufficient,
            degraded=sufficient and drop is not None and drop + 1e-12 >= self.policy.max_f1_drop,
        )
