import json
from datetime import UTC, datetime

import pytest

from src.agents.diagnosis.contracts import DiagnosisInput
from src.agents.diagnosis.evaluation.evaluate import load_scenarios, render_markdown, run_evaluation
from src.agents.diagnosis.llm_analyzer import (
    DEFAULT_LLM_BASE_URL,
    ClaudeSupplementalAnalyzer,
    LLMConfigurationError,
    OpenAICompatibleAnalyzer,
    build_analyzer_from_env,
    retry_delay_seconds,
)
from src.agents.diagnosis.result_builder import build_result
from src.agents.diagnosis.tools import static_toolbox


GROQ = "https://api.groq.com/openai/v1"
ROLLOUT_REF = "EVD-INC-LLM-KUBERNETES-DEPLOYMENT_HISTORY-01"
ROLLOUT = {"kubernetes": {"deployment_history": [{
    "evidence_type": "k8s_event",
    "summary": "Deployment revision 42 rolled out.",
    "value": {"change_id": "payment-service-rev42", "description": "Deployment revision 42",
              "changed_at": "2026-10-20T11:40:00+00:00"},
}]}}


class FakeTransport:
    """Replays (status, headers, body) responses and records every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, url, body, headers, timeout):
        self.requests.append({"url": url, "body": json.loads(json.dumps(body)), "headers": headers})
        status, response_headers, response = self.responses.pop(0)
        return status, response_headers, response if isinstance(response, str) else json.dumps(response)


def _final(selected, refs, content=None):
    answer = {"summary": "Reviewed.", "selected_cause": selected, "evidence_refs": refs,
              "limitations": ["Heap profile missing."]}
    text = content if content is not None else json.dumps(answer)
    return 200, {}, {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": text}}]}


def _tool_call(name, operation):
    call = {"id": "call_1", "type": "function",
            "function": {"name": name, "arguments": json.dumps({"operation": operation})}}
    return 200, {}, {"choices": [{"finish_reason": "tool_calls",
                                  "message": {"role": "assistant", "content": None, "tool_calls": [call]}}]}


def _memory_input():
    return DiagnosisInput(
        incident_id="INC-LLM",
        service_id="payment-service",
        namespace="prod",
        environment="production",
        issue_summary="Memory rising after a release.",
        signals={"memory_percent": 91, "restart_count": 2},
        memory_rise_started_at=datetime(2026, 10, 20, 12, tzinfo=UTC),
    )


def test_tool_calls_collect_evidence_that_the_rules_reevaluate():
    transport = FakeTransport(
        _tool_call("kubernetes", "deployment_history"),
        _final("MEMORY_LEAK_POST_DEPLOY", [ROLLOUT_REF]),
    )
    analyzer = OpenAICompatibleAnalyzer(
        GROQ, "llama-3.3-70b-versatile", "gsk-test", toolbox=static_toolbox(ROLLOUT), transport=transport
    )

    result = build_result(_memory_input(), analyzer=analyzer)

    assert result.primary_root_cause == "MEMORY_LEAK_POST_DEPLOY"
    assert [(call.tool, call.requested_by) for call in result.tool_calls] == [("kubernetes", "llm")]
    assert result.supplemental_analysis.source == "llm"
    first, second = transport.requests
    assert first["url"] == GROQ + "/chat/completions"
    assert first["headers"]["Authorization"] == "Bearer gsk-test"
    assert first["headers"]["User-Agent"] == "pids-diagnosis-agent/1.0"  # urllib's default gets 403/1010
    assert first["body"]["model"] == "llama-3.3-70b-versatile"
    tool = first["body"]["tools"][0]
    assert tool["type"] == "function" and tool["function"]["parameters"]["required"] == ["operation"]
    assert "MEMORY_PRESSURE" in first["body"]["messages"][0]["content"]  # schema in the system prompt
    assert second["body"]["messages"][-1] == {
        "role": "tool", "tool_call_id": "call_1", "content": second["body"]["messages"][-1]["content"]
    }
    assert ROLLOUT_REF in second["body"]["messages"][-1]["content"]
    assert any("llama-3.3-70b-versatile" in item for item in result.limitations)


def test_ollama_without_key_sends_no_authorization_and_last_round_forces_json():
    transport = FakeTransport(_tool_call("kubernetes", "deployment_history"), _final("NONE", []))
    analyzer = OpenAICompatibleAnalyzer(
        DEFAULT_LLM_BASE_URL, "qwen2.5:7b", toolbox=static_toolbox(ROLLOUT),
        transport=transport, max_tool_rounds=1,
    )

    analyzer.analyze(_memory_input(), *_review_inputs())

    first, last = transport.requests
    assert "Authorization" not in first["headers"]
    assert "tools" in first["body"] and "response_format" not in first["body"]
    assert "tools" not in last["body"] and last["body"]["response_format"] == {"type": "json_object"}


def _review_inputs():
    baseline = build_result(_memory_input())
    return tuple(baseline.hypotheses), baseline.evidence, baseline.knowledge


def test_rate_limit_waits_for_the_server_delay_then_retries():
    waits = []
    transport = FakeTransport(
        (429, {"Retry-After": "7"}, "rate limited"),
        (429, {}, '{"error": {"details": [{"retryDelay": "12s"}]}}'),
        (503, {}, "overloaded"),
        _final("MEMORY_PRESSURE", []),
    )
    analyzer = OpenAICompatibleAnalyzer(
        GROQ, "m", "k", transport=transport, sleep=waits.append, retry_base_seconds=2
    )

    analysis = analyzer.analyze(_memory_input(), *_review_inputs())

    assert analysis.selected_cause == "MEMORY_PRESSURE"
    assert waits == [7.0, 12.0, 8.0]  # header, Gemini body hint, exponential backoff (2 * 2**2)
    assert analyzer.retries == 3


def test_exhausted_retries_fall_back_to_the_rules():
    transport = FakeTransport(*[(429, {"retry-after": "1"}, "slow down")] * 3)
    analyzer = OpenAICompatibleAnalyzer(GROQ, "m", "k", transport=transport, sleep=lambda _: None, max_retries=2)

    result = build_result(_memory_input(), analyzer=analyzer)

    assert result.primary_root_cause == "MEMORY_PRESSURE"
    assert result.supplemental_analysis is None
    assert any("HTTP 429 still returned after 2 retries" in item for item in result.limitations)


def _tool_use_failed(generation):
    error = {"message": "attempted to call tool 'json' which was not in request.tools",
             "type": "invalid_request_error", "code": "tool_use_failed", "failed_generation": generation}
    return 400, {}, {"error": error}


def test_answer_sent_as_a_made_up_json_tool_call_is_recovered():
    answer = {"summary": "OK.", "selected_cause": "MEMORY_PRESSURE", "evidence_refs": [], "limitations": []}
    generation = json.dumps({"name": "json", "arguments": answer})
    analyzer = OpenAICompatibleAnalyzer(GROQ, "openai/gpt-oss-120b", "k",
                                        transport=FakeTransport(_tool_use_failed(generation)))

    analysis = analyzer.analyze(_memory_input(), *_review_inputs())

    assert analysis.source == "llm" and analysis.selected_cause == "MEMORY_PRESSURE"


def test_unrecoverable_tool_failure_asks_once_for_plain_json_without_tools():
    transport = FakeTransport(_tool_use_failed('{"name": "kubernetes"}'), _final("MEMORY_PRESSURE", []))
    analyzer = OpenAICompatibleAnalyzer(GROQ, "m", "k", toolbox=static_toolbox(ROLLOUT), transport=transport)

    assert analyzer.analyze(_memory_input(), *_review_inputs()).selected_cause == "MEMORY_PRESSURE"
    retry = transport.requests[1]["body"]
    assert "tools" not in retry and retry["messages"][-1]["content"].startswith("Do not call any tool")

    twice = FakeTransport(_tool_use_failed("{}"), _tool_use_failed("{}"))
    failing = OpenAICompatibleAnalyzer(GROQ, "m", "k", toolbox=static_toolbox(ROLLOUT), transport=twice)
    assert build_result(_memory_input(), analyzer=failing).supplemental_analysis is None


def test_llm_rejecting_every_candidate_flags_the_diagnosis_for_review():
    rejection = {"summary": "Heap is flat; the release only lowered the limit, so no leak.",
                 "selected_cause": "NONE", "evidence_refs": [], "limitations": []}
    analyzer = OpenAICompatibleAnalyzer(GROQ, "m", "k", transport=FakeTransport(
        _final(None, None, json.dumps(rejection))))

    result = build_result(_memory_input(), analyzer=analyzer)

    assert result.primary_root_cause == "MEMORY_PRESSURE"  # the rules keep their ranking
    assert result.requires_additional_evidence is True
    assert any("found no candidate supported" in item and "lowered the limit" in item
               for item in result.limitations)


def test_repeated_llm_runs_report_mean_worst_case_and_timeouts():
    s01 = next(item for item in load_scenarios() if item.scenario_id.startswith("S01"))
    # Same incident labelled with another cause: the rules are then wrong and sure of themselves.
    mislabelled = s01.model_copy(update={
        "scenario_id": "S99-mislabelled",
        "expected": s01.expected.model_copy(update={"root_cause": "DEPENDENCY_UNAVAILABLE"}),
    })
    scenarios = [s01, mislabelled]

    def timed_out(*_):
        raise TimeoutError("The read operation timed out")

    def factory(toolbox):
        return OpenAICompatibleAnalyzer(GROQ, "m", "k", toolbox=toolbox, transport=timed_out)

    report = run_evaluation(scenarios, factory, "openai-compatible:m", llm_runs=2)

    assert len(report["llm_runs"]) == 2
    assert all(run["llm_timeouts"] == 2 and run["avoided_confident_errors"] == 0 for run in report["llm_runs"])
    assert report["metrics"]["rules+tools+qdrant"]["confident_errors"] == 1  # the mislabelled copy
    markdown = render_markdown(report)
    assert "| **Pire cas** |" in markdown and "0 à 0 erreur(s) sûre(s) d’elle(s) évitée(s) sur 2" in markdown
    assert "| 1 sur 2 |" in markdown


def test_retry_delay_parsing():
    assert retry_delay_seconds({"retry-after": "3.5"}, "", 0, 2) == 3.5
    assert retry_delay_seconds({}, "", 3, 2) == 16
    assert retry_delay_seconds({"Retry-After": "Thu, 01 Jan 1970 00:00:00 GMT"}, "", 0, 2) == 0.0


def test_fenced_json_is_accepted_and_invented_evidence_ids_are_dropped():
    fenced = "```json\n" + json.dumps(
        {"summary": "OK.", "selected_cause": "MEMORY_PRESSURE",
         "evidence_refs": ["EVD-MADE-UP"], "limitations": []}
    ) + "\n```"
    analyzer = OpenAICompatibleAnalyzer(GROQ, "m", "k", transport=FakeTransport(_final(None, None, fenced)))

    analysis = analyzer.analyze(_memory_input(), *_review_inputs())

    assert analysis.selected_cause == "MEMORY_PRESSURE"
    assert analysis.evidence_refs == ()
    assert any("EVD-MADE-UP" in item for item in analysis.limitations)


def test_cause_outside_candidates_or_unreachable_endpoint_fall_back():
    invalid = OpenAICompatibleAnalyzer(GROQ, "m", "k", transport=FakeTransport(_final("DISK_FULL", [])))
    assert build_result(_memory_input(), analyzer=invalid).supplemental_analysis is None

    def refused(*_):
        raise ConnectionRefusedError("connection refused")

    offline = OpenAICompatibleAnalyzer(DEFAULT_LLM_BASE_URL, "m", transport=refused)
    result = build_result(_memory_input(), analyzer=offline)
    assert any("cannot reach http://localhost:11434/v1" in item for item in result.limitations)


def test_environment_configuration():
    assert build_analyzer_from_env(env={}) is None
    with pytest.raises(LLMConfigurationError, match="LLM_MODEL"):
        build_analyzer_from_env(env={}, provider="openai")
    with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY"):
        build_analyzer_from_env(env={"LLM_MODEL": "m", "LLM_API_KEY_ENV": "GROQ_API_KEY"}, provider="openai")

    groq = build_analyzer_from_env(
        env={"PIDS_DIAGNOSIS_LLM": "openai", "LLM_BASE_URL": GROQ, "LLM_MODEL": "llama-3.3-70b-versatile",
             "LLM_API_KEY_ENV": "GROQ_API_KEY", "GROQ_API_KEY": "gsk-secret", "LLM_MAX_RETRIES": "3"}
    )
    assert isinstance(groq, OpenAICompatibleAnalyzer)
    assert groq._api_key == "gsk-secret" and groq._max_retries == 3
    assert groq.model_label == f"openai-compatible:llama-3.3-70b-versatile @ {GROQ}"

    ollama = build_analyzer_from_env(env={"LLM_MODEL": "qwen2.5:7b"}, provider="openai")
    assert ollama._base_url == DEFAULT_LLM_BASE_URL and ollama._api_key is None

    claude = build_analyzer_from_env(env={}, provider="anthropic")
    assert isinstance(claude, ClaudeSupplementalAnalyzer) and claude.model_label.startswith("anthropic:")


def test_evaluation_reports_the_model_and_waits_between_llm_scenarios():
    scenarios = load_scenarios()[:3]
    waits = []

    def factory(toolbox):
        return OpenAICompatibleAnalyzer(
            GROQ, "llama-3.3-70b-versatile", "k", toolbox=toolbox,
            transport=FakeTransport(*[(429, {}, "")] * 2), sleep=lambda _: None, max_retries=1,
        )

    report = run_evaluation(scenarios, factory, "openai-compatible:llama-3.3-70b-versatile", 4.0, waits.append)

    assert waits == [4.0, 4.0]  # between the three LLM scenarios only
    assert report["models"]["rules"] == "aucun (règles déterministes)"
    assert report["models"]["full+llm"] == "openai-compatible:llama-3.3-70b-versatile"
    metrics = report["metrics"]["full+llm"]
    assert metrics["llm_answer_rate"] == 0 and metrics["llm_retries"] == 3
    assert report["metrics"]["rules"]["llm_answer_rate"] is None
    markdown = render_markdown(report)
    assert "| `full+llm` | openai-compatible:llama-3.3-70b-versatile |" in markdown
    assert "(LLM non utilisé)" in markdown and "Délai entre les scénarios LLM : 4.0 s." in markdown
