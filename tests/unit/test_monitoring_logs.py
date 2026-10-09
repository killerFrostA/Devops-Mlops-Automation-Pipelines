import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr

from src.agents.monitoring.collectors.loki import LokiClient
from src.agents.monitoring.collectors.models import LogLine
from src.agents.monitoring.detectors.logs import (
    FindingKind,
    LabelVerdict,
    LogDetector,
    LogFinding,
    LogLabel,
    LogPatternDetector,
    LogPolicy,
    redact,
    regex_label,
)
from src.agents.monitoring.labelers.gemini import GeminiLabeler

NOW = 1_000_000.0
POLICY = LogPolicy()  # 30 min baseline, 5 min recent window


def lines(text: str, count: int, start: float) -> list[LogLine]:
    return [LogLine(start + i, text.format(i=i)) for i in range(count)]


def baseline() -> list[LogLine]:
    """Normal history: access logs plus a known, rare warning."""
    start = NOW - POLICY.recent_seconds - POLICY.baseline_seconds
    return lines('INFO 172.18.0.1 - "GET /pay HTTP/1.1" 200 OK', 500, start) + lines(
        "WARNING slow query took {i}ms on orders", 6, start + 600
    )


def recent(*extra: list[LogLine]) -> list[LogLine]:
    normal = lines('INFO 172.18.0.1 - "GET /pay HTTP/1.1" 200 OK', 100, NOW - 290)
    return normal + [line for block in extra for line in block]


def run(detector: LogPatternDetector, base: list[LogLine], rec: list[LogLine]) -> list[LogFinding]:
    return asyncio.run(detector.evaluate(base, rec))


class FakeLabeler:
    def __init__(self, verdict: LabelVerdict | None) -> None:
        self.verdict = verdict
        self.calls: list[str] = []

    async def label(self, template: str) -> LabelVerdict | None:
        self.calls.append(template)
        return self.verdict


def test_normal_logs_produce_no_finding() -> None:
    assert run(LogPatternDetector(), baseline(), recent()) == []


def test_new_connection_refused_pattern_is_reported_as_a_fact() -> None:
    refused = lines("ERROR Connection refused to payments-db:5432 request_id=r{i}", 340, NOW - 200)
    findings = run(LogPatternDetector(), baseline(), recent(refused))
    assert len(findings) == 1
    finding = findings[0]
    assert finding.kind is FindingKind.NEW_PATTERN
    assert finding.label is LogLabel.CONNECTION_REFUSED
    assert finding.label_source == "regex"
    assert finding.count_recent == 340
    assert finding.count_baseline == 0
    assert "Connection refused to payments-db:5432" in finding.template
    assert "<*>" in finding.template  # the varying request id became a wildcard


def test_known_warning_surging_is_reported_as_surge() -> None:
    surge = lines("WARNING slow query took {i}ms on orders", 60, NOW - 250)
    findings = run(LogPatternDetector(), baseline(), recent(surge))
    assert [f.kind for f in findings] == [FindingKind.SURGE]
    assert findings[0].count_baseline == 6


def test_known_warning_at_usual_rate_is_ignored() -> None:
    usual = lines("WARNING slow query took {i}ms on orders", 1, NOW - 250)
    assert run(LogPatternDetector(), baseline(), recent(usual)) == []


def test_a_few_new_errors_are_below_min_count() -> None:
    few = lines("ERROR Connection refused to payments-db:5432 request_id=r{i}", 3, NOW - 100)
    assert run(LogPatternDetector(), baseline(), recent(few)) == []


def test_unlabelled_template_without_llm_is_unknown_and_needs_review() -> None:
    odd = lines("ERROR Ledger reconciliation mismatch batch={i}", 20, NOW - 100)
    finding = run(LogPatternDetector(), baseline(), recent(odd))[0]
    assert finding.label is LogLabel.UNKNOWN
    assert finding.label_source == "none"
    assert finding.requires_review


def test_llm_is_only_asked_when_regex_cannot_label() -> None:
    labeler = FakeLabeler(LabelVerdict(LogLabel.HTTP_SERVER_ERROR, 0.9, requires_review=False))
    odd = lines("ERROR Ledger reconciliation mismatch batch={i}", 20, NOW - 100)
    refused = lines("ERROR Connection refused to payments-db:5432 request_id=r{i}", 20, NOW - 100)
    findings = run(LogPatternDetector(labeler), baseline(), recent(odd, refused))
    by_source = {f.label_source: f for f in findings}
    assert by_source["llm"].label is LogLabel.HTTP_SERVER_ERROR
    assert by_source["regex"].label is LogLabel.CONNECTION_REFUSED
    assert len(labeler.calls) == 1


def test_llm_budget_and_cache_bound_the_calls() -> None:
    labeler = FakeLabeler(LabelVerdict(LogLabel.UNKNOWN, 0.9, requires_review=True))
    detector = LogPatternDetector(labeler, LogPolicy(max_llm_calls=2))
    # Different word counts, so Drain3 keeps five distinct templates instead of merging them.
    blocks = [lines("ERROR ledger " + "step " * n + "code=c{i}", 10, NOW - 100) for n in range(5)]
    run(detector, baseline(), recent(*blocks))
    assert len(labeler.calls) == 2
    run(detector, baseline(), recent(*blocks))  # cached verdicts are reused, budget refilled
    assert len(labeler.calls) == 4


def test_secrets_are_redacted_before_mining() -> None:
    text = (
        "ERROR login failed password=hunter2 token: abc.def Bearer eyJhbGci "
        "user=alice@example.com from 10.0.0.12 session=0123456789abcdef0123"
    )
    cleaned = redact(text)
    for secret in (
        "hunter2",
        "abc.def",
        "eyJhbGci",
        "alice@example.com",
        "10.0.0.12",
        "0123456789abcdef0123",
    ):
        assert secret not in cleaned
    leaked = lines(text, 10, NOW - 100)
    finding = run(LogPatternDetector(), baseline(), recent(leaked))[0]
    assert "hunter2" not in finding.template and "hunter2" not in finding.example


@pytest.mark.parametrize(
    ("template", "label"),
    [
        ("ERROR Timeout after <*> calling payment-gateway", LogLabel.TIMEOUT),
        ("ERROR java.lang.OutOfMemoryError: Java heap space", LogLabel.OUT_OF_MEMORY),
        ("ERROR Invalid token for user <*>", LogLabel.AUTHENTICATION_ERROR),
        ("WARNING Readiness probe failed: <*>", LogLabel.READINESS_FAILURE),
        ('INFO "POST /pay HTTP/1.1" 503 Service Unavailable', LogLabel.HTTP_SERVER_ERROR),
        ("ERROR Ledger mismatch", None),
    ],
)
def test_regex_taxonomy(template: str, label: LogLabel | None) -> None:
    assert regex_label(template) is label


def gemini_with(answer: object, status: int = 200) -> tuple[GeminiLabeler, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if status != 200:
            return httpx.Response(status)
        text = json.dumps(answer)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]})

    http = httpx.AsyncClient(base_url="https://gemini", transport=httpx.MockTransport(handler))
    return GeminiLabeler(SecretStr("test-key"), http=http), seen


def test_gemini_valid_answer_is_accepted_and_request_is_safe() -> None:
    labeler, seen = gemini_with({"label": "TIMEOUT", "confidence": 0.92})
    verdict = asyncio.run(labeler.label("ERROR request stalled password=hunter2 <*>"))
    assert verdict == LabelVerdict(LogLabel.TIMEOUT, 0.92, requires_review=False)
    request = seen[0]
    assert request.headers["x-goog-api-key"] == "test-key"
    assert "hunter2" not in request.content.decode()
    assert "gemini-2.5-flash:generateContent" in request.url.path


@pytest.mark.parametrize(
    "answer",
    [
        {"label": "DATABASE_DOWN", "confidence": 0.99},  # invented label, outside the taxonomy
        {"label": "TIMEOUT", "confidence": 0.3},  # too unsure
        {"label": "TIMEOUT", "confidence": 7},  # nonsense confidence
        ["TIMEOUT"],  # not an object
    ],
)
def test_gemini_untrustworthy_answers_become_unknown(answer: object) -> None:
    labeler, _ = gemini_with(answer)
    verdict = asyncio.run(labeler.label("ERROR odd event"))
    assert verdict is None or (verdict.label is LogLabel.UNKNOWN and verdict.requires_review)


def test_gemini_failure_returns_none_so_the_agent_keeps_working() -> None:
    labeler, _ = gemini_with({}, status=503)
    assert asyncio.run(labeler.label("ERROR odd event")) is None


def test_gemini_model_name_is_validated() -> None:
    with pytest.raises(ValueError, match="model"):
        GeminiLabeler(SecretStr("k"), model="../../evil")


def test_log_detector_reads_loki_and_splits_windows() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["query"] == '{service_id="payment-service"}'
        start_ns = int(request.url.params["start"])
        values = [[str(start_ns + i * 1_000_000_000), "INFO ok"] for i in range(50)]
        recent_ns = int((NOW - 100) * 1_000_000_000)
        values += [
            [str(recent_ns + i), f"ERROR Connection refused to db:5432 id={i}"] for i in range(30)
        ]
        streams = [{"stream": {"service_id": "payment-service"}, "values": values}]
        return httpx.Response(
            200, json={"status": "success", "data": {"resultType": "streams", "result": streams}}
        )

    async def go() -> list[LogFinding]:
        http = httpx.AsyncClient(base_url="http://loki", transport=httpx.MockTransport(handler))
        async with LokiClient("http://loki", http=http) as client:
            return await LogDetector(client, clock=lambda: NOW).evaluate("payment-service")

    findings = asyncio.run(go())
    assert [(f.kind, f.label, f.count_recent) for f in findings] == [
        (FindingKind.NEW_PATTERN, LogLabel.CONNECTION_REFUSED, 30)
    ]
