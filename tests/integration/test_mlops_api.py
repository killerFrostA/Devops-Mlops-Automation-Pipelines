import asyncio
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from src.api.app import create_app
from src.config import Settings
from src.contracts.mlops import ModelHealthReport

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "contracts/examples/agent6/model-health-input.json"


def send(body):
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=create_app(Settings(environment="test", _env_file=None))
            ),
            base_url="http://test",
        ) as client:
            return await client.post("/api/v1/mlops/health/assess", json=body)

    return asyncio.run(request())


def test_assessment_api_returns_valid_diagnostic_report():
    response = send(json.loads(EXAMPLE.read_text(encoding="utf-8")))
    assert response.status_code == 200
    report = ModelHealthReport.model_validate(response.json())
    assert report.assessment.decision == "RETRAIN_AND_EVALUATE"
    assert report.performance.current == pytest.approx(2 / 3)
    assert report.assessment.drift_score == max(item.score for item in report.distributions)


@pytest.mark.parametrize("problem", ["missing_fields", "small_window", "unaligned_labels"])
def test_api_rejects_invalid_assessment_input(problem):
    body = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    if problem == "missing_fields":
        body.pop("model_name")
    elif problem == "small_window":
        body["distributions"][0]["current_counts"] = [1] * 4
    else:
        body["performance"]["labels"] = [1]
    assert send(body).status_code == 422


def test_cli_runs_example_with_explicit_policy():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.agents.mlops_lifecycle",
            "--input",
            str(EXAMPLE),
            "--policy",
            str(ROOT / "configs/agent6-health-policy.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    report = ModelHealthReport.model_validate_json(result.stdout)
    assert report.assessment.decision == "RETRAIN_AND_EVALUATE"
    assert report.policy.version == "agent6-health-v1"
