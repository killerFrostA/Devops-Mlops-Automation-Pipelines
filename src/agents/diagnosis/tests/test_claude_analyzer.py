import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from src.agents.diagnosis.contracts import DiagnosisInput, EvidenceItem
from src.agents.diagnosis.llm_analyzer import (
    DEFAULT_CLAUDE_MODEL,
    ClaudeSupplementalAnalyzer,
    build_analyzer_from_env,
)
from src.agents.diagnosis.result_builder import build_result
from src.agents.diagnosis.tools import static_toolbox
from src.contracts.enums import Environment


class FakeMessages:
    def __init__(self, responses: tuple[SimpleNamespace, ...]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> SimpleNamespace:
        self.requests.append(request)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, *responses: SimpleNamespace) -> None:
        self.beta = SimpleNamespace(messages=FakeMessages(responses))
        self.messages = self.beta.messages


def _final(selected: str, refs: list[str], summary: str = "Reviewed.") -> SimpleNamespace:
    answer = {
        "summary": summary,
        "selected_cause": selected,
        "evidence_refs": refs,
        "limitations": ["Heap profile missing."],
    }
    return SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=json.dumps(answer)),
        ],
    )


def _tool_call(name: str, operation: str) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(
                type="tool_use", id="toolu_1", name=name, input={"operation": operation}
            )
        ],
    )


def _memory_input() -> DiagnosisInput:
    return DiagnosisInput(
        incident_id="INC-LLM",
        service_id="payment-service",
        namespace="prod",
        environment=Environment.PRODUCTION,
        issue_summary="Memory rising after a release.",
        signals={"memory_percent": 91, "restart_count": 2},
        memory_rise_started_at=datetime(2026, 10, 20, 12, tzinfo=UTC),
    )


ROLLOUT = {
    "kubernetes": {
        "deployment_history": [
            {
                "evidence_type": "k8s_event",
                "summary": "Deployment revision 42 rolled out.",
                "value": {
                    "change_id": "payment-service-rev42",
                    "description": "Deployment revision 42",
                    "changed_at": "2026-10-20T11:40:00+00:00",
                },
            }
        ]
    }
}


def test_claude_collects_evidence_with_tools_and_rules_reevaluate_it() -> None:
    toolbox = static_toolbox(ROLLOUT)
    client = FakeClient(
        _tool_call("kubernetes", "deployment_history"),
        _final("MEMORY_LEAK_POST_DEPLOY", ["EVD-INC-LLM-KUBERNETES-DEPLOYMENT_HISTORY-01"]),
    )
    analyzer = ClaudeSupplementalAnalyzer(client, toolbox=toolbox)

    result = build_result(_memory_input(), analyzer=analyzer)

    # Without the collected rollout the rules only see MEMORY_PRESSURE.
    assert result.primary_root_cause == "MEMORY_LEAK_POST_DEPLOY"
    assert result.recommended_actions[0].action == "ROLLBACK_OR_REVERT_DEPLOYMENT"
    assert [(call.tool, call.requested_by) for call in result.tool_calls] == [("kubernetes", "llm")]
    assert result.supplemental_analysis is not None
    assert result.supplemental_analysis.source == "llm"
    first, second = client.beta.messages.requests
    assert first["model"] == DEFAULT_CLAUDE_MODEL
    assert first["betas"] == ["server-side-fallback-2026-07-01"] and first["fallbacks"] == "default"
    assert first["output_config"]["effort"] == "high"
    schema = first["output_config"]["format"]["schema"]
    assert schema["properties"]["selected_cause"]["enum"] == ["MEMORY_PRESSURE", "NONE"]
    assert first["tools"][0]["strict"] is True
    tool_result = second["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "toolu_1" and tool_result["is_error"] is False


def _two_candidates() -> DiagnosisInput:
    # Memory pressure and CPU pressure come out as close rule candidates.
    return DiagnosisInput(
        incident_id="INC-RANK",
        service_id="payment-service",
        namespace="prod",
        environment=Environment.PRODUCTION,
        issue_summary="Memory and CPU pressure.",
        signals={"memory_percent": 88, "cpu_percent": 70, "restart_count": 3, "latency_ms": 850},
    )


def test_claude_may_reorder_close_candidates_only() -> None:
    baseline = build_result(_two_candidates())
    top, second = baseline.hypotheses[0], baseline.hypotheses[1]
    assert top.probability - second.probability <= 0.15

    reranked = build_result(
        _two_candidates(),
        analyzer=ClaudeSupplementalAnalyzer(
            FakeClient(_final(second.code, list(second.evidence_refs)))
        ),
    )

    assert reranked.primary_root_cause == second.code
    assert any("ranking was adjusted" in item for item in reranked.limitations)


def test_large_disagreement_keeps_rules_and_requires_review() -> None:
    payload = DiagnosisInput(
        incident_id="INC-DISAGREE",
        service_id="checkout-api",
        namespace="prod",
        environment=Environment.PRODUCTION,
        issue_summary="Errors.",
        signals={"error_rate": 0.25},
        evidence=[
            EvidenceItem(
                evidence_id="LOG",
                evidence_type="log",
                source="loki",
                summary="connection refused",
                value={"lines": ["connection refused"]},
            )
        ],
    )
    baseline = build_result(payload)
    far = baseline.hypotheses[-1]
    assert baseline.hypotheses[0].probability - far.probability > 0.15

    result = build_result(
        payload, analyzer=ClaudeSupplementalAnalyzer(FakeClient(_final(far.code, [])))
    )

    assert result.primary_root_cause == baseline.primary_root_cause
    assert result.requires_additional_evidence is True
    assert any("disagreement needs review" in item for item in result.limitations)


def test_refusal_invalid_cause_and_missing_sdk_fall_back_to_rules() -> None:
    refusal = SimpleNamespace(stop_reason="refusal", content=[])
    refused = build_result(
        _memory_input(), analyzer=ClaudeSupplementalAnalyzer(FakeClient(refusal))
    )
    invented = build_result(
        _memory_input(), analyzer=ClaudeSupplementalAnalyzer(FakeClient(_final("ALIENS", [])))
    )

    for result in (refused, invented):
        assert result.primary_root_cause == "MEMORY_PRESSURE"
        assert result.supplemental_analysis is None
        assert any("Supplemental analysis failed" in item for item in result.limitations)


def test_no_cause_answer_and_fallbacks_can_be_disabled() -> None:
    client = FakeClient(_final("NONE", []))
    analysis = ClaudeSupplementalAnalyzer(client, use_fallbacks=False).analyze(
        _memory_input(),
        tuple(build_result(_memory_input()).hypotheses),
        (),
        build_result(_memory_input()).knowledge,
    )

    assert analysis.selected_cause is None
    assert "betas" not in client.messages.requests[0]


def test_claude_is_enabled_only_on_explicit_opt_in() -> None:
    assert build_analyzer_from_env(env={}) is None
    analyzer = build_analyzer_from_env(
        env={"PIDS_DIAGNOSIS_LLM": "claude", "PIDS_DIAGNOSIS_LLM_EFFORT": "medium"}
    )
    assert isinstance(analyzer, ClaudeSupplementalAnalyzer)
    assert analyzer._effort == "medium" and analyzer._model == DEFAULT_CLAUDE_MODEL
