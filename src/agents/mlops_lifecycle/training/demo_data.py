"""Seeded, clearly synthetic machine-failure data for local lifecycle verification."""

import math
from datetime import UTC, datetime, timedelta
from random import Random

from src.contracts.mlops_training import TrainingDataset, TrainingObservation


def synthetic_machine_training_dataset(seed: int = 42, rows: int = 600) -> TrainingDataset:
    if rows < 300:
        raise ValueError("The lifecycle demo needs at least 300 synthetic observations")
    rng = Random(seed)
    start = datetime(2026, 9, 20, tzinfo=UTC)
    observations = []
    machine_types = ("press", "conveyor", "robot")
    for index in range(rows):
        temperature = round(rng.uniform(40, 105), 3)
        vibration = round(rng.uniform(0, 12), 3)
        machine_type = machine_types[index % len(machine_types)]
        risk = -5.3 + 0.018 * (temperature - 70) + 0.78 * vibration
        risk += 0.25 if machine_type == "robot" else 0.0
        failure_probability = 1 / (1 + math.exp(-risk))
        observations.append(
            TrainingObservation(
                observation_id=f"synthetic-{index:04}",
                group_id=f"machine-{index:04}",
                observed_at=start + timedelta(minutes=index),
                features={
                    "temperature_c": temperature,
                    "vibration_mm_s": vibration,
                    "machine_type": machine_type,
                },
                label=int(rng.random() < failure_probability),
            )
        )
    return TrainingDataset(
        dataset_ref=f"synthetic-machine-training-seed-{seed}",
        model_name="machine-failure",
        observations=tuple(observations),
    )
