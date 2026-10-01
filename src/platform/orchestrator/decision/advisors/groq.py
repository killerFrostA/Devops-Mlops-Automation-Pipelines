"""Optional Groq candidate-ranking advisor. It has no action authority."""

from pathlib import Path
from typing import Annotated

import httpx
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.platform.orchestrator.decision.contracts import AdvisorRequest, AdvisorSuggestion
from src.platform.orchestrator.decision.prompts.renderer import render_candidate_ranking_prompt

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
ORCHESTRATOR_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class GroqAdvisorSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATOR_GROQ_",
        env_file=ORCHESTRATOR_ENV_FILE,
        extra="ignore",
    )

    enabled: bool = False
    api_key: SecretStr = SecretStr("")
    model: Annotated[str, Field(min_length=1)] = "openai/gpt-oss-20b"
    timeout_seconds: Annotated[float, Field(gt=0, le=30)] = 3.0

    @model_validator(mode="after")
    def require_key_when_enabled(self) -> "GroqAdvisorSettings":
        if self.enabled and not self.api_key.get_secret_value():
            raise ValueError("Groq advisor is enabled but its API key is missing")
        return self


class GroqDecisionAdvisor:
    """Send only typed candidate summaries, never raw evidence or action parameters."""

    def __init__(
        self,
        settings: GroqAdvisorSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not settings.enabled:
            raise ValueError("Groq advisor must be explicitly enabled")
        self._settings = settings
        self._transport = transport

    async def suggest(self, incoming: AdvisorRequest) -> AdvisorSuggestion:
        request = AdvisorRequest.model_validate(incoming.model_dump(mode="json"))
        candidate_ids = [candidate.action_id for candidate in request.candidates]
        schema = {
            "type": "object",
            "properties": {
                "candidate_action_id": {"type": "string", "enum": candidate_ids},
                "reason_code": {"type": "string"},
            },
            "required": ["candidate_action_id", "reason_code"],
            "additionalProperties": False,
        }
        system_prompt, user_prompt = render_candidate_ranking_prompt(request)
        body = {
            "model": self._settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "candidate_preference",
                    "strict": True,
                    "schema": schema,
                },
            },
            "temperature": 0,
            "max_completion_tokens": 120,
            "stream": False,
        }
        async with httpx.AsyncClient(
            timeout=self._settings.timeout_seconds, transport=self._transport
        ) as client:
            response = await client.post(
                GROQ_CHAT_COMPLETIONS_URL,
                headers={
                    "Authorization": f"Bearer {self._settings.api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
        response.raise_for_status()
        if len(response.content) > 16_384:
            raise ValueError("Groq response exceeds the permitted size")
        document = response.json()
        choices = document.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("Groq returned an invalid completion count")
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
            raise ValueError("Groq completion did not finish normally")
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("refusal"):
            raise ValueError("Groq did not return a usable message")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Groq returned no structured content")
        return AdvisorSuggestion.model_validate_json(content)


def configured_groq_advisor(
    settings: GroqAdvisorSettings | None = None,
) -> GroqDecisionAdvisor | None:
    """Create the optional advisor only when explicitly enabled."""
    configured = settings or GroqAdvisorSettings()
    if not configured.enabled:
        return None
    return GroqDecisionAdvisor(configured)
