"""Offline stage boundaries; no model or dataset is bundled."""

from src.platform.errors import IntegrationNotConfiguredError


def prepare_dataset(dataset_ref: str) -> None:
    raise IntegrationNotConfiguredError("Implement versioned data validation and split generation")


def train(dataset_ref: str, experiment_name: str) -> None:
    raise IntegrationNotConfiguredError("Implement reproducible training and tracked artifacts")


def evaluate(candidate_ref: str, champion_ref: str, dataset_ref: str) -> None:
    raise IntegrationNotConfiguredError(
        "Implement champion/challenger evaluation and quality gates"
    )
