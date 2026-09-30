"""Mocked provider transport: no API credentials or network traffic required."""

import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest

from src.agents.mlops_lifecycle.llm_judge import ResponsesAPIJudge
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
