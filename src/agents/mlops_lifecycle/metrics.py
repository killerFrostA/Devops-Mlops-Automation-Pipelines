"""Deterministic distribution and binary classification diagnostics."""

import math

from src.contracts.mlops import DistributionComparison, LabeledPredictionWindow


def population_stability_index(comparison: DistributionComparison, epsilon: float = 1e-6) -> float:
    """Add epsilon to every bin proportion, normalize, then sum symmetric divergence."""

    if not math.isfinite(epsilon) or not 1e-12 <= epsilon <= 0.01:
        raise ValueError("epsilon must be finite and between 1e-12 and 0.01")
    reference_total = sum(comparison.reference_counts)
    current_total = sum(comparison.current_counts)
    normalization = 1 + len(comparison.bins) * epsilon
    terms = []
    for reference_count, current_count in zip(
        comparison.reference_counts, comparison.current_counts, strict=True
    ):
        reference = (reference_count / reference_total + epsilon) / normalization
        current = (current_count / current_total + epsilon) / normalization
        terms.append((current - reference) * (math.log(current) - math.log(reference)))
    return math.fsum(terms)


def binary_confusion_counts(window: LabeledPredictionWindow) -> tuple[int, int, int, int, int]:
    """Return TP, FP, FN, labeled count and positive-label count; ignore missing labels."""

    tp = fp = fn = labeled = positives = 0
    for prediction, label in zip(window.predictions, window.labels, strict=True):
        if label is None:
            continue
        labeled += 1
        positives += label
        tp += int(prediction == 1 and label == 1)
        fp += int(prediction == 1 and label == 0)
        fn += int(prediction == 0 and label == 1)
    return tp, fp, fn, labeled, positives
