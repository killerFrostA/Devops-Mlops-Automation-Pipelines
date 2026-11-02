from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
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


def build_analyzer_from_env(
    toolbox: Toolbox | None = None,
    env: Mapping[str, str] | None = None,
) -> ClaudeSupplementalAnalyzer | None:
    """Enable Claude only on explicit opt-in (PIDS_DIAGNOSIS_LLM=claude): calls cost money."""
    env = os.environ if env is None else env
    if env.get("PIDS_DIAGNOSIS_LLM", "").casefold() not in ("claude", "anthropic"):
        return None
    return ClaudeSupplementalAnalyzer(
        model=env.get("PIDS_DIAGNOSIS_LLM_MODEL") or DEFAULT_CLAUDE_MODEL,
        effort=env.get("PIDS_DIAGNOSIS_LLM_EFFORT") or DEFAULT_LLM_EFFORT,
        toolbox=toolbox,
        use_fallbacks=env.get("PIDS_DIAGNOSIS_LLM_FALLBACKS", "true").casefold() != "false",
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
