"""Run the local Agent 6 machine-failure lifecycle with synthetic training data."""

import argparse
import asyncio
from pathlib import Path

from src.agents.mlops_lifecycle.demo_data import synthetic_machine_training_dataset
from src.agents.mlops_lifecycle.local_files import (
    FileBaselineRepository,
    FileLabelRepository,
    FilePredictionRepository,
    LocalSourceConfig,
)
from src.agents.mlops_lifecycle.mlflow_registry import MLflowModelRegistry
from src.agents.mlops_lifecycle.service import MLOpsLifecycleAgent
from src.agents.mlops_lifecycle.training import SklearnTrainingPipeline
from src.contracts.mlops_training import TrainingPolicy

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 6 synthetic machine-failure lifecycle")
    parser.add_argument(
        "--sources",
        type=Path,
        default=ROOT / "contracts/examples/agent6/local-sources/machine-sources.json",
    )
    parser.add_argument(
        "--training-policy", type=Path, default=ROOT / "configs/agent6-training-demo.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/agent6-demo")
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        source = LocalSourceConfig.model_validate_json(args.sources.read_text(encoding="utf-8"))
        policy = TrainingPolicy.model_validate_json(
            args.training_policy.read_text(encoding="utf-8")
        )
        base = args.sources.resolve().parent
        agent = MLOpsLifecycleAgent(
            baselines=FileBaselineRepository(base / source.baseline_file),
            predictions=FilePredictionRepository(base / source.predictions_file),
            labels=FileLabelRepository(base / source.labels_file),
            training=SklearnTrainingPipeline(args.output_dir),
            registry=MLflowModelRegistry(args.output_dir),
        )
        dataset = synthetic_machine_training_dataset(seed=args.seed, rows=args.rows)
        report = asyncio.run(agent.assess_and_train(source.query, dataset, policy))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "lifecycle-report.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
