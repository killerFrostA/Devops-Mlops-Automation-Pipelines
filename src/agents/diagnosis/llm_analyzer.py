from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

from src.agents.diagnosis.contracts import (
    DiagnosisInput,
    EvidenceItem,
    Hypothesis,
    KnowledgeRetrieval,
    SupplementalAnalysis,
    ToolCallRecord,
)
from src.agents.diagnosis.investigation import tool_context
from src.agents.diagnosis.tools import Toolbox


DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_LLM_EFFORT = "high"
DEFAULT_LLM_BASE_URL = "http://localhost:11434/v1"  # Ollama: local, no API key.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
USER_AGENT = "pids-diagnosis-agent/1.0"
NO_CAUSE = "NONE"
_MAX_EVIDENCE_VALUE_CHARS = 600

SYSTEM_PROMPT = """You review a root-cause analysis for a DevOps incident.

Deterministic rules have already ranked candidate root causes from the evidence. Your job:
1. Check whether the evidence really supports the leading candidate, or a close alternative.
2. When an observation would change the ranking, collect it with the read-only evidence tools.
   Call a tool only for that reason; each tool operation returns evidence with an ID.
3. Answer with the JSON format requested.

Constraints:
- selected_cause must be one of the candidate codes, or NONE if no candidate is supported.
- evidence_refs may only contain evidence IDs given to you or returned by the tools.
- Historical incidents and runbooks are context, never evidence about this incident.
- You cannot change any system; do not propose commands.
- summary: at most three sentences stating the conclusion and the evidence IDs behind it.
- limitations: what is missing or uncertain, one short sentence each."""


class LLMAnalysisError(RuntimeError):
    """The language-model analysis could not produce a usable answer."""


class _ToolUseFailed(LLMAnalysisError):
    """The endpoint rejected the model's tool call (Groq ``tool_use_failed``)."""

    def __init__(self, message: str, failed_generation: str) -> None:
        super().__init__(message)
        self.failed_generation = failed_generation


class LLMConfigurationError(ValueError):
    """The LLM environment variables are missing or inconsistent."""


def _compact(value: Any) -> Any:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value if len(text) <= _MAX_EVIDENCE_VALUE_CHARS else text[:_MAX_EVIDENCE_VALUE_CHARS] + "..."


def _incident_prompt(
    payload: DiagnosisInput,
    hypotheses: tuple[Hypothesis, ...],
    evidence: tuple[EvidenceItem, ...],
    knowledge: KnowledgeRetrieval,
) -> str:
    document = {
        "incident": {
            "incident_id": payload.incident_id,
            "service_id": payload.service_id,
            "namespace": payload.namespace,
            "environment": payload.environment.value,
            "issue_summary": payload.issue_summary,
            "signals": _compact(
                {key: value for key, value in payload.signals.items() if key != "agent1_context"}
            ),
            "recent_changes": [
                {"change_id": event.change_id, "description": event.description,
                 "changed_at": event.changed_at.isoformat()}
                for event in payload.recent_change_events
            ],
        },
        "candidates": [
            {"rank": item.rank, "code": item.code, "rule_score": item.probability,
             "evidence_refs": list(item.evidence_refs)}
            for item in hypotheses
        ],
        "evidence": [
            {"evidence_id": item.evidence_id, "type": item.evidence_type, "source": item.source,
             "provenance": item.provenance, "summary": item.summary, "value": _compact(item.value)}
            for item in evidence
        ],
        "similar_past_incidents": [
            {"reference_id": item.reference_id, "root_cause": item.root_cause, "summary": item.summary}
            for item in knowledge.similar_incidents
        ],
        "runbooks": [
            {"reference_id": item.reference_id, "summary": item.summary} for item in knowledge.runbooks
        ],
    }
    return (
        "Review this diagnosis and answer in the requested JSON format.\n\n"
        + json.dumps(document, ensure_ascii=False, indent=1, sort_keys=True, default=str)
    )


def _answer_schema(candidate_codes: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "selected_cause": {"type": "string", "enum": [*candidate_codes, NO_CAUSE]},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
            "limitations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "selected_cause", "evidence_refs", "limitations"],
        "additionalProperties": False,
    }


class ClaudeSupplementalAnalyzer:
    """Supplemental RCA review by Claude, with read-only evidence tools.

    The answer is advisory: result_builder validates the selected cause against the rule
    candidates and the cited evidence IDs, and re-runs the rules on any evidence the model
    collected. Without the ``anthropic`` package or credentials the call fails and the
    diagnosis continues with the deterministic rules only.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        model: str = DEFAULT_CLAUDE_MODEL,
        effort: str = DEFAULT_LLM_EFFORT,
        toolbox: Toolbox | None = None,
        max_tool_rounds: int = 4,
        max_tokens: int = 16000,
        timeout_seconds: float = 120.0,
        use_fallbacks: bool = True,
    ) -> None:
        self._client = client
        self._model = model
        self._effort = effort
        self._toolbox = toolbox
        self._max_tool_rounds = max_tool_rounds
        self._max_tokens = max_tokens
        self._timeout_seconds = timeout_seconds
        self._use_fallbacks = use_fallbacks

    @property
    def model_label(self) -> str:
        return f"anthropic:{self._model}"

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise LLMAnalysisError("the 'anthropic' package is not installed") from exc
            self._client = anthropic.Anthropic(timeout=self._timeout_seconds)
        return self._client

    def _create(self, **request: Any) -> Any:
        client = self._get_client()
        if self._use_fallbacks:
            # On a policy refusal the API re-runs the request on a fallback model.
            return client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
            )
        return client.messages.create(**request)

    def _run_tools(
        self,
        blocks: list[Any],
        payload: DiagnosisInput,
        deadline: datetime,
        collected: list[EvidenceItem],
        records: list[ToolCallRecord],
    ) -> list[dict[str, Any]]:
        context = tool_context(payload, deadline)
        results = []
        for block in blocks:
            arguments = block.input if isinstance(block.input, Mapping) else {}
            operation = arguments.get("operation")
            if self._toolbox is None or not isinstance(operation, str):
                content, is_error = "Error: unknown tool or missing operation.", True
            else:
                result = self._toolbox.call(block.name, operation, context, requested_by="llm")
                records.append(result.record)
                known = {item.evidence_id for item in collected}
                collected.extend(item for item in result.evidence if item.evidence_id not in known)
                is_error = result.record.status != "succeeded"
                content = json.dumps(
                    {
                        "status": result.record.status,
                        "error": result.record.error,
                        "evidence": [
                            {"evidence_id": item.evidence_id, "type": item.evidence_type,
                             "summary": item.summary, "value": _compact(item.value)}
                            for item in result.evidence
                        ],
                    },
                    ensure_ascii=False,
                    default=str,
                )
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error}
            )
        return results

    def analyze(
        self,
        payload: DiagnosisInput,
        hypotheses: tuple[Hypothesis, ...],
        evidence: tuple[EvidenceItem, ...],
        knowledge: KnowledgeRetrieval,
    ) -> SupplementalAnalysis:
        candidate_codes = [item.code for item in hypotheses]
        deadline = datetime.now(UTC) + timedelta(seconds=self._timeout_seconds)
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": SYSTEM_PROMPT,
            "output_config": {
                "effort": self._effort,
                "format": {"type": "json_schema", "schema": _answer_schema(candidate_codes)},
            },
        }
        if self._toolbox is not None:
            request["tools"] = self._toolbox.definitions()
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": _incident_prompt(payload, hypotheses, evidence, knowledge)}
        ]
        collected: list[EvidenceItem] = []
        records: list[ToolCallRecord] = []

        for _ in range(self._max_tool_rounds + 1):
            if datetime.now(UTC) >= deadline:
                raise LLMAnalysisError("analysis deadline reached")
            response = self._create(messages=messages, **request)
            if response.stop_reason == "refusal":
                raise LLMAnalysisError("the model declined the request")
            if response.stop_reason == "max_tokens":
                raise LLMAnalysisError("the answer was truncated at max_tokens")
            if response.stop_reason in ("tool_use", "pause_turn"):
                # Keep the whole assistant turn (thinking blocks included) unchanged.
                messages.append({"role": "assistant", "content": response.content})
                tool_blocks = [block for block in response.content if block.type == "tool_use"]
                if tool_blocks:
                    messages.append(
                        {"role": "user", "content": self._run_tools(
                            tool_blocks, payload, deadline, collected, records
                        )}
                    )
                continue
            text = next((block.text for block in response.content if block.type == "text"), None)
            if text is None:
                raise LLMAnalysisError("the answer contained no text block")
            try:
                answer = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMAnalysisError("the answer was not valid JSON") from exc
            selected = answer.get("selected_cause")
            return SupplementalAnalysis(
                source="llm",
                summary=answer.get("summary") or "No summary returned.",
                selected_cause=None if selected == NO_CAUSE else selected,
                evidence_refs=tuple(dict.fromkeys(answer.get("evidence_refs", []))),
                limitations=(
                    *answer.get("limitations", []),
                    f"Supplemental review by {self._model}; validated against the rule "
                    "candidates and unable to execute actions.",
                ),
                collected_evidence=tuple(collected),
                tool_calls=tuple(records),
            )
        raise LLMAnalysisError(f"no final answer after {self._max_tool_rounds} tool rounds")


Transport = Callable[[str, dict[str, Any], dict[str, str], float], tuple[int, Mapping[str, str], str]]


def _urllib_transport(
    url: str, body: dict[str, Any], headers: dict[str, str], timeout: float
) -> tuple[int, Mapping[str, str], str]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read().decode("utf-8", errors="replace")


def retry_delay_seconds(headers: Mapping[str, str], body: str, attempt: int, base: float) -> float:
    """Wait asked by the server (Retry-After, Gemini retryDelay), else exponential backoff."""
    retry_after = next((v for k, v in headers.items() if k.casefold() == "retry-after"), None)
    if retry_after:
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            try:
                return max((parsedate_to_datetime(retry_after) - datetime.now(UTC)).total_seconds(), 0.0)
            except (TypeError, ValueError):
                pass
    hinted = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', body)
    if hinted:
        return float(hinted.group(1))
    return base * 2**attempt


def _json_answer(text: str) -> dict[str, Any]:
    """Parse the JSON answer, tolerating Markdown fences or text around it."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        answer = json.loads(cleaned)
    except json.JSONDecodeError:
        first, last = cleaned.find("{"), cleaned.rfind("}")
        try:
            answer = json.loads(cleaned[first : last + 1]) if 0 <= first < last else None
        except json.JSONDecodeError:
            answer = None
    if not isinstance(answer, dict):
        raise LLMAnalysisError("the answer was not a valid JSON object")
    return answer


PLAIN_JSON_REMINDER = (
    "Do not call any tool now. Reply with the final JSON object as plain message text."
)


def _answer_from_failed_generation(text: str) -> str | None:
    """Return the answer JSON hidden in a rejected tool call, if there is one."""
    try:
        generation = _json_answer(text)
    except LLMAnalysisError:
        return None
    for candidate in (generation, generation.get("arguments"), generation.get("parameters")):
        if isinstance(candidate, str):
            try:
                candidate = _json_answer(candidate)
            except LLMAnalysisError:
                continue
        if isinstance(candidate, dict) and "selected_cause" in candidate:
            return json.dumps(candidate)
    return None


class OpenAICompatibleAnalyzer:
    """Supplemental RCA review through any OpenAI-compatible Chat Completions endpoint.

    Works with Groq, Gemini (OpenAI compatibility layer) and Ollama, using only the standard
    library. Rate limits (429) and transient server errors are retried after the delay the
    server asks for. Like the Claude analyzer, the answer is advisory and validated by
    result_builder.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        toolbox: Toolbox | None = None,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_tool_rounds: int = 4,
        max_tokens: int = 4000,
        timeout_seconds: float = 300.0,
        max_retries: int = 5,
        retry_base_seconds: float = 2.0,
        retry_max_seconds: float = 90.0,
        json_mode: bool = True,
    ) -> None:
        self._base_url = base_url
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._api_key = api_key
        self._toolbox = toolbox
        self._transport = transport or _urllib_transport
        self._sleep = sleep
        self._max_tool_rounds = max_tool_rounds
        self._max_tokens = max_tokens
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_seconds = retry_max_seconds
        self._json_mode = json_mode
        self.retries = 0

    @property
    def model_label(self) -> str:
        return f"openai-compatible:{self._model} @ {self._base_url}"

    def _tools(self) -> list[dict[str, Any]]:
        assert self._toolbox is not None
        return [
            {
                "type": "function",
                "function": {
                    "name": item["name"],
                    "description": item["description"],
                    "parameters": item["input_schema"],
                },
            }
            for item in self._toolbox.definitions()
        ]

    def _post(self, body: dict[str, Any], deadline: datetime) -> dict[str, Any]:
        # Cloudflare-fronted endpoints (Groq) reject urllib's default User-Agent with 403/1010.
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        for attempt in range(self._max_retries + 1):
            remaining = (deadline - datetime.now(UTC)).total_seconds()
            if remaining <= 0:
                raise LLMAnalysisError("analysis deadline reached")
            try:
                status, response_headers, text = self._transport(self._url, body, headers, remaining)
            except OSError as exc:  # URLError, refused connection, timeout.
                raise LLMAnalysisError(f"cannot reach {self._base_url}: {exc}") from exc
            if status in RETRYABLE_STATUS and attempt < self._max_retries:
                wait = min(
                    retry_delay_seconds(response_headers, text, attempt, self._retry_base_seconds),
                    self._retry_max_seconds,
                )
                if (deadline - datetime.now(UTC)).total_seconds() <= wait:
                    raise LLMAnalysisError(f"HTTP {status}: no time left to wait {wait:.0f}s and retry")
                self.retries += 1
                self._sleep(wait)
                continue
            if status in RETRYABLE_STATUS:
                raise LLMAnalysisError(f"HTTP {status} still returned after {self._max_retries} retries")
            if status == 400:
                try:
                    error = json.loads(text).get("error") or {}
                except (json.JSONDecodeError, AttributeError):
                    error = {}
                if isinstance(error, dict) and error.get("code") == "tool_use_failed":
                    raise _ToolUseFailed(str(error.get("message", "")), str(error.get("failed_generation", "")))
            if status != 200:
                raise LLMAnalysisError(f"HTTP {status} from {self._base_url}: {' '.join(text.split())[:300]}")
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMAnalysisError("the endpoint returned invalid JSON") from exc
        raise AssertionError("unreachable")

    def _run_tools(
        self,
        tool_calls: list[dict[str, Any]],
        payload: DiagnosisInput,
        deadline: datetime,
        collected: list[EvidenceItem],
        records: list[ToolCallRecord],
    ) -> list[dict[str, Any]]:
        context = tool_context(payload, deadline)
        messages = []
        for call in tool_calls:
            function = call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            operation = arguments.get("operation") if isinstance(arguments, Mapping) else None
            if self._toolbox is None or not isinstance(operation, str):
                content = "Error: unknown tool or missing operation."
            else:
                result = self._toolbox.call(function.get("name", ""), operation, context, requested_by="llm")
                records.append(result.record)
                known = {item.evidence_id for item in collected}
                collected.extend(item for item in result.evidence if item.evidence_id not in known)
                content = json.dumps(
                    {
                        "status": result.record.status,
                        "error": result.record.error,
                        "evidence": [
                            {"evidence_id": item.evidence_id, "type": item.evidence_type,
                             "summary": item.summary, "value": _compact(item.value)}
                            for item in result.evidence
                        ],
                    },
                    ensure_ascii=False,
                    default=str,
                )
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": content})
        return messages

    def analyze(
        self,
        payload: DiagnosisInput,
        hypotheses: tuple[Hypothesis, ...],
        evidence: tuple[EvidenceItem, ...],
        knowledge: KnowledgeRetrieval,
    ) -> SupplementalAnalysis:
        candidate_codes = [item.code for item in hypotheses]
        deadline = datetime.now(UTC) + timedelta(seconds=self._timeout_seconds)
        # Compatible endpoints do not all enforce a JSON schema: it is given in the prompt and
        # the answer is checked here.
        system = (
            SYSTEM_PROMPT
            + "\n\nAnswer with one JSON object only, matching this JSON schema:\n"
            + json.dumps(_answer_schema(candidate_codes))
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": _incident_prompt(payload, hypotheses, evidence, knowledge)},
        ]
        collected: list[EvidenceItem] = []
        records: list[ToolCallRecord] = []

        corrected = False
        for round_index in range(self._max_tool_rounds + 1):
            body: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": self._max_tokens,
                "temperature": 0,
            }
            # The last round offers no tools, so the model has to give its final answer.
            if self._toolbox is not None and round_index < self._max_tool_rounds and not corrected:
                body["tools"] = self._tools()
            elif self._json_mode:
                body["response_format"] = {"type": "json_object"}
            try:
                response = self._post(body, deadline)
            except _ToolUseFailed as exc:
                # Some models (gpt-oss) emit the final answer as a call to a made-up "json" tool.
                recovered = _answer_from_failed_generation(exc.failed_generation)
                if recovered is not None:
                    response = {"choices": [{"finish_reason": "stop", "message": {"content": recovered}}]}
                elif not corrected:
                    corrected = True
                    messages.append({"role": "user", "content": PLAIN_JSON_REMINDER})
                    continue
                else:
                    raise
            try:
                choice = response["choices"][0]
                message = choice["message"]
            except (KeyError, IndexError, TypeError) as exc:
                raise LLMAnalysisError("the response contained no choice") from exc
            if choice.get("finish_reason") == "length":
                raise LLMAnalysisError("the answer was truncated at max_tokens")
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                messages.append(
                    {"role": "assistant", "content": message.get("content"), "tool_calls": tool_calls}
                )
                messages.extend(self._run_tools(tool_calls, payload, deadline, collected, records))
                continue
            if not message.get("content"):
                raise LLMAnalysisError("the answer contained no text")
            answer = _json_answer(message["content"])
            # result_builder checks the cause against the candidates re-ranked with the
            # collected evidence.
            selected = answer.get("selected_cause")
            # Smaller models sometimes invent evidence IDs; drop them instead of the whole review.
            known_refs = {item.evidence_id for item in (*evidence, *collected)}
            cited = [str(ref) for ref in dict.fromkeys(answer.get("evidence_refs") or [])]
            invented = [ref for ref in cited if ref not in known_refs]
            limitations = [str(item) for item in answer.get("limitations") or []]
            if invented:
                limitations.append("Ignored evidence IDs unknown to this diagnosis: " + ", ".join(invented))
            return SupplementalAnalysis(
                source="llm",
                summary=answer.get("summary") or "No summary returned.",
                selected_cause=None if selected == NO_CAUSE else selected,
                evidence_refs=tuple(ref for ref in cited if ref in known_refs),
                limitations=(
                    *limitations,
                    f"Supplemental review by {self.model_label}; validated against the rule "
                    "candidates and unable to execute actions.",
                ),
                collected_evidence=tuple(collected),
                tool_calls=tuple(records),
            )
        raise LLMAnalysisError(f"no final answer after {self._max_tool_rounds} tool rounds")


_DISABLED = ("", "0", "false", "off", "none", "no")
_ANTHROPIC = ("claude", "anthropic")


def _number_env(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise LLMConfigurationError(f"{name} must be a number, got {raw!r}") from exc


def build_analyzer_from_env(
    toolbox: Toolbox | None = None,
    env: Mapping[str, str] | None = None,
    provider: str | None = None,
) -> ClaudeSupplementalAnalyzer | OpenAICompatibleAnalyzer | None:
    """Build the LLM reviewer, on explicit opt-in only.

    ``provider`` (else PIDS_DIAGNOSIS_LLM) selects it. ``openai`` — the default once enabled —
    calls an OpenAI-compatible endpoint configured by LLM_BASE_URL, LLM_MODEL and
    LLM_API_KEY_ENV (the *name* of the variable holding the key; leave it unset for Ollama).
    ``anthropic`` calls Claude (ANTHROPIC_API_KEY, billed).
    """
    env = os.environ if env is None else env
    choice = (provider if provider is not None else env.get("PIDS_DIAGNOSIS_LLM", "")).strip().casefold()
    if choice in _DISABLED:
        return None
    if choice in _ANTHROPIC:
        return ClaudeSupplementalAnalyzer(
            model=env.get("PIDS_DIAGNOSIS_LLM_MODEL") or DEFAULT_CLAUDE_MODEL,
            effort=env.get("PIDS_DIAGNOSIS_LLM_EFFORT") or DEFAULT_LLM_EFFORT,
            toolbox=toolbox,
            use_fallbacks=env.get("PIDS_DIAGNOSIS_LLM_FALLBACKS", "true").casefold() != "false",
        )

    model = env.get("LLM_MODEL", "").strip()
    if not model:
        raise LLMConfigurationError("LLM_MODEL is not set (for example llama-3.3-70b-versatile on Groq).")
    key_variable = env.get("LLM_API_KEY_ENV", "").strip()
    api_key = None
    if key_variable:
        api_key = env.get(key_variable, "").strip()
        if not api_key:
            raise LLMConfigurationError(
                f"LLM_API_KEY_ENV names {key_variable}, but that variable is empty or not set."
            )
    return OpenAICompatibleAnalyzer(
        base_url=env.get("LLM_BASE_URL", "").strip() or DEFAULT_LLM_BASE_URL,
        model=model,
        api_key=api_key,
        toolbox=toolbox,
        max_retries=int(_number_env(env, "LLM_MAX_RETRIES", 5)),
        retry_base_seconds=_number_env(env, "LLM_RETRY_BASE_SECONDS", 2.0),
        timeout_seconds=_number_env(env, "LLM_TIMEOUT_SECONDS", 300.0),
        json_mode=env.get("LLM_JSON_MODE", "true").strip().casefold() != "false",
    )


def analyze_with_llm(
    payload: DiagnosisInput,
    hypotheses: tuple[Hypothesis, ...],
    evidence: tuple[EvidenceItem, ...],
    knowledge: KnowledgeRetrieval,
) -> SupplementalAnalysis:
    """Return a reproducible supplement; this function does not call an LLM."""

    best = hypotheses[0] if hypotheses else None
    known_refs = {item.evidence_id for item in evidence}
    evidence_refs = tuple(ref for ref in (best.evidence_refs if best else ()) if ref in known_refs)

    summary = (
        f"Local deterministic analyzer reviewed incident {payload.incident_id} using "
        f"{len(evidence)} supplied or derived evidence "
        f"items and {len(knowledge.similar_incidents)} similar-incident fixtures plus "
        f"{len(knowledge.runbooks)} runbook fixtures."
    )
    if best is not None and best.code != "INSUFFICIENT_EVIDENCE":
        summary += f" It agrees with the rule-ranked candidate {best.code}."
    else:
        summary += " It cannot select a supported root cause from the available evidence."

    limitations = [
        "This is a deterministic local simulation, not a language-model response.",
        "Historical fixtures are context only and are not evidence about the active incident.",
        "The analyzer cannot execute infrastructure actions.",
    ]
    if not evidence_refs:
        limitations.append("No current-incident evidence reference supports the leading candidate.")

    return SupplementalAnalysis(
        source="deterministic_simulation",
        summary=summary,
        selected_cause=best.code if best is not None else None,
        evidence_refs=evidence_refs,
        limitations=tuple(limitations),
    )


class DeterministicSupplementalAnalyzer:
    """Offline implementation of the future supplemental-analyzer port."""

    def analyze(
        self,
        payload: DiagnosisInput,
        hypotheses: tuple[Hypothesis, ...],
        evidence: tuple[EvidenceItem, ...],
        knowledge: KnowledgeRetrieval,
    ) -> SupplementalAnalysis:
        return analyze_with_llm(payload, hypotheses, evidence, knowledge)
