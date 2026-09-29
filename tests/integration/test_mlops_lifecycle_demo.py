"""A real local Agent 6 run with MLflow tracking and candidate registry."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def test_machine_demo_tracks_and_registers_candidate(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    sklearn = pytest.importorskip("sklearn")
    assert sklearn.__version__
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.agents.mlops_lifecycle.demo",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(completed.stdout)
    assert report["health"]["assessment"]["decision"] == "RETRAIN_AND_EVALUATE"
    assert report["decision"] == "REQUEST_PROMOTION_REVIEW"
    assert report["registered_model_version"] == "1"
    assert report["evaluation"]["candidate"]["f1"] > report["evaluation"]["champion"]["f1"]
    assert report["evaluation"]["candidate"]["recall"] >= report["evaluation"]["champion"]["recall"]
    assert (tmp_path / "mlflow.db").exists()
    assert (tmp_path / "lifecycle-report.json").exists()
    assert (tmp_path / "mlruns").exists()

    mlflow.set_tracking_uri("sqlite:///" + (tmp_path / "mlflow.db").as_posix())
    version = mlflow.MlflowClient().get_model_version("machine-failure", "1")
    assert version.source
    assert version.tags["review_status"] == "PENDING"
    assert version.tags["dataset_sha256"] == report["evaluation"]["dataset_sha256"]
    run = mlflow.get_run(report["evaluation"]["run_id"])
    assert run.data.params["dataset_sha256"] == report["evaluation"]["dataset_sha256"]
    assert run.data.metrics["candidate_test_f1"] == report["evaluation"]["candidate"]["f1"]
