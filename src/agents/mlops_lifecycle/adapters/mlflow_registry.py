"""Local MLflow registry adapter for reviewed candidate versions."""

from pathlib import Path

from src.contracts.mlops_training import CandidateEvaluation


class MLflowModelRegistry:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()

    async def register_candidate(self, evaluation: CandidateEvaluation) -> str:
        import mlflow

        mlflow.set_tracking_uri("sqlite:///" + (self.output_dir / "mlflow.db").as_posix())
        version = mlflow.register_model(
            evaluation.candidate_model_uri,
            evaluation.model_name,
            tags={
                "agent": "mlops-lifecycle",
                "review_status": "PENDING",
                "run_id": evaluation.run_id,
                "dataset_ref": evaluation.dataset_ref,
                "dataset_sha256": evaluation.dataset_sha256,
                "policy_version": evaluation.policy.version,
                "pipeline_version": evaluation.pipeline_version,
            },
        )
        return str(version.version)
