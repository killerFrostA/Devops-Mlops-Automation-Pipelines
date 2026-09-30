"""Mocked provider transport: no API credentials or network traffic required."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from src.agents.mlops_lifecycle.evaluation.llm_judge import LLMJudgeSettings, ResponsesAPIJudge
from src.contracts.mlops_projects import TextEvaluationSample


def sample() -> TextEvaluationSample:
    return TextEvaluationSample(
        observation_id="item-1",
        observed_at=datetime(2026, 9, 30, tzinfo=UTC),
        prompt="What is 2+2?",
        output="4",
        reference="4",
        rubric="Score correctness",
    )


@pytest.mark.parametrize(
    ("provider", "hostname"),
    [("openai", "api.openai.com"), ("groq", "api.groq.com")],
)
def test_provider_uses_structured_output_and_validates_result(provider: str, hostname: str) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": '{"score":0.85,"rationale":"Correct"}'}
                        ],
                    }
                ],
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            judge = ResponsesAPIJudge(provider, "test-model", "dummy", client=client)
            result = await judge.judge(sample())
            assert result.score == 0.85

    asyncio.run(run())
    assert requests[0].url.host == hostname
    payload = json.loads(requests[0].content)
    assert payload["store"] is False
    assert payload["text"]["format"]["strict"] is True
    assert payload["input"].find("What is 2+2?") >= 0


@pytest.mark.parametrize(
    "body",
    [
        {"status": "incomplete", "output": []},
        {"status": "completed", "output": [{"type": "message", "content": None}]},
        {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]},
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"score":9,"rationale":"bad"}'}],
                }
            ],
        },
    ],
)
def test_provider_fails_closed_on_incomplete_refused_or_invalid_output(body: dict) -> None:
    async def run() -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
        async with httpx.AsyncClient(transport=transport) as client:
            judge = ResponsesAPIJudge("openai", "test-model", "dummy", client=client)
            with pytest.raises(RuntimeError, match="no score|invalid|refused|malformed"):
                await judge.judge(sample())

    asyncio.run(run())


def test_provider_http_error_has_no_score_or_key_in_error() -> None:
    async def run() -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(429, json={"error": "rate"}))
        async with httpx.AsyncClient(transport=transport) as client:
            judge = ResponsesAPIJudge("groq", "test-model", "secret-test-key", client=client)
            with pytest.raises(RuntimeError, match="no score") as exc:
                await judge.judge(sample())
            assert "secret-test-key" not in str(exc.value)

    asyncio.run(run())


def test_local_env_file_loads_provider_model_and_key_with_process_override(
    tmp_path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AGENT6_LLM_PROVIDER=groq\nAGENT6_LLM_MODEL=test-model\nGROQ_API_KEY=from-local-file\n",
        encoding="utf-8",
    )
    for name in ("AGENT6_LLM_PROVIDER", "AGENT6_LLM_MODEL", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    settings = LLMJudgeSettings(_env_file=env_file)
    assert settings.agent6_llm_provider == "groq"
    assert settings.agent6_llm_model == "test-model"
    assert settings.api_key_for("groq") == "from-local-file"
    judge = ResponsesAPIJudge.from_env("groq", settings.agent6_llm_model, settings=settings)
    assert judge.model_name == "groq:test-model"

    monkeypatch.setenv("GROQ_API_KEY", "from-process")
    overridden = LLMJudgeSettings(_env_file=env_file)
    assert overridden.api_key_for("groq") == "from-process"


def test_default_llm_settings_use_agent_package_env_file() -> None:
    repository = Path(__file__).resolve().parents[2]
    expected = repository / "src/agents/mlops_lifecycle/.env"
    assert LLMJudgeSettings.model_config["env_file"] == expected
