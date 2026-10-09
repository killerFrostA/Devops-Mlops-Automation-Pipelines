"""Log analysis: group error lines into templates (Drain3), label them, report what is new.

Pipeline: redact secrets -> keep error lines -> Drain3 templates -> closed-taxonomy label
(regex first, optional LLM only for templates regex cannot label) -> compare the recent window
with the baseline window.

A finding is an observation ("340 occurrences of a new CONNECTION_REFUSED template"), never a
cause: the label describes the message, not why it happened. That is the Diagnosis agent's job.
"""

import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from pydantic import BaseModel, ConfigDict, Field

from src.agents.monitoring.collectors.models import LogLine
from src.contracts.enums import Severity


class LogLabel(StrEnum):
    """Closed taxonomy. A labeler may only pick one of these; anything else is UNKNOWN."""

    CONNECTION_REFUSED = "CONNECTION_REFUSED"
    TIMEOUT = "TIMEOUT"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    HTTP_SERVER_ERROR = "HTTP_SERVER_ERROR"
    READINESS_FAILURE = "READINESS_FAILURE"
    UNKNOWN = "UNKNOWN"


# Ordered: the first matching rule wins (an OOM message may also mention a 500).
REGEX_RULES: tuple[tuple[LogLabel, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in (
        (LogLabel.OUT_OF_MEMORY, r"out of memory|memoryerror|oom-?kill|cannot allocate memory"),
        (LogLabel.CONNECTION_REFUSED, r"connection refused|econnrefused"),
        (LogLabel.TIMEOUT, r"timed? ?out|deadline exceeded"),
        (
            LogLabel.AUTHENTICATION_ERROR,
            r"unauthori[sz]ed|authentication failed|invalid (token|credentials)|forbidden",
        ),
        (LogLabel.READINESS_FAILURE, r"readiness|liveness|health ?check failed|not ready"),
        (LogLabel.HTTP_SERVER_ERROR, r"internal server error|unhandled exception|\b5\d\d\b"),
    )
)

ERROR_LINE = re.compile(r"\b(ERROR|CRITICAL|FATAL|WARN(ING)?)\b|Traceback|Exception")

# Applied before anything else: secrets never reach Drain3 templates, findings or an LLM.
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)\b(password|passwd|pwd|token|secret|api[_-]?key|authorization)\b(\s*[:=]\s*)\S+"
        ),
        r"\1\2<redacted>",
    ),
    (re.compile(r"(?i)\bbearer\s+\S+"), "Bearer <redacted>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<email>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),
    (re.compile(r"\b[0-9a-fA-F]{16,}\b"), "<hex>"),
)


def redact(text: str) -> str:
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def regex_label(template: str) -> LogLabel | None:
    for label, pattern in REGEX_RULES:
        if pattern.search(template):
            return label
    return None


@dataclass(frozen=True, slots=True)
class LabelVerdict:
    label: LogLabel
    confidence: float
    requires_review: bool


class LogLabeler(Protocol):
    """Optional LLM fallback. Returns None when it cannot answer (no key, error, timeout)."""

    async def label(self, template: str) -> LabelVerdict | None: ...


class LogPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_seconds: float = Field(default=1800, gt=0)
    recent_seconds: float = Field(default=300, gt=0)
    min_count: int = Field(default=5, ge=1)  # fewer occurrences are noise, not a pattern
    surge_factor: float = Field(default=5.0, gt=1)  # known template, rate x5 vs baseline
    max_llm_calls: int = Field(default=10, ge=0)  # cost bound per evaluation


class FindingKind(StrEnum):
    NEW_PATTERN = "NEW_PATTERN"
    SURGE = "SURGE"


@dataclass(frozen=True, slots=True)
class LogFinding:
    kind: FindingKind
    template: str
    label: LogLabel
    label_source: str  # "regex", "llm" or "none"
    count_recent: int
    count_baseline: int
    example: str  # one redacted recent line
    requires_review: bool
    severity: Severity = Severity.MEDIUM


def _new_miner() -> TemplateMiner:
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    return TemplateMiner(config=config)


class LogPatternDetector:
    def __init__(self, labeler: LogLabeler | None = None, policy: LogPolicy | None = None):
        self._labeler = labeler
        self.policy = policy or LogPolicy()
        self._llm_cache: dict[str, LabelVerdict] = {}
        self._llm_budget = self.policy.max_llm_calls

    async def evaluate(
        self, baseline: Sequence[LogLine], recent: Sequence[LogLine]
    ) -> list[LogFinding]:
        miner = _new_miner()  # stateless per run: same input, same templates

        def mine(lines: Sequence[LogLine]) -> tuple[Counter[int], dict[int, str]]:
            counts: Counter[int] = Counter()
            examples: dict[int, str] = {}
            for line in lines:
                text = redact(line.text)
                if not ERROR_LINE.search(text):
                    continue
                cluster_id = miner.add_log_message(text)["cluster_id"]
                counts[cluster_id] += 1
                examples.setdefault(cluster_id, text)
            return counts, examples

        baseline_counts, _ = mine(baseline)
        recent_counts, examples = mine(recent)
        templates = {c.cluster_id: c.get_template() for c in miner.drain.clusters}
        ratio = self.policy.baseline_seconds / self.policy.recent_seconds

        findings: list[LogFinding] = []
        self._llm_budget = self.policy.max_llm_calls
        for cluster_id, count in recent_counts.most_common():
            if count < self.policy.min_count:
                continue
            before = baseline_counts.get(cluster_id, 0)
            if before == 0:
                kind = FindingKind.NEW_PATTERN
            elif count * ratio >= self.policy.surge_factor * before:
                kind = FindingKind.SURGE
            else:
                continue
            template = templates[cluster_id]
            label, source, review = regex_label(template), "regex", False
            if label is None:
                verdict = await self._ask_llm(template)
                if verdict is None:
                    label, source, review = LogLabel.UNKNOWN, "none", True
                else:
                    label, source, review = verdict.label, "llm", verdict.requires_review
            findings.append(
                LogFinding(
                    kind=kind,
                    template=template,
                    label=label,
                    label_source=source,
                    count_recent=count,
                    count_baseline=before,
                    example=examples[cluster_id],
                    requires_review=review,
                )
            )
        return findings

    async def _ask_llm(self, template: str) -> LabelVerdict | None:
        """Cached LLM label. Failures are not cached, so a later run can retry them."""
        if template in self._llm_cache:
            return self._llm_cache[template]
        if self._labeler is None or self._llm_budget <= 0:
            return None
        self._llm_budget -= 1
        verdict = await self._labeler.label(template)
        if verdict is not None:
            self._llm_cache[template] = verdict
        return verdict


class LogSource(Protocol):
    async def query_lines(
        self, service_id: str, start: float, end: float, limit: int = 5000
    ) -> tuple[LogLine, ...]: ...


class LogDetector:
    """Fetches baseline + recent lines from a log source, then runs the pattern detector."""

    def __init__(
        self,
        source: LogSource,
        patterns: LogPatternDetector | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._source = source
        self._patterns = patterns or LogPatternDetector()
        self._clock = clock

    async def evaluate(self, service_id: str) -> list[LogFinding]:
        policy = self._patterns.policy
        now = self._clock()
        recent_start = now - policy.recent_seconds
        lines = await self._source.query_lines(
            service_id, recent_start - policy.baseline_seconds, now
        )
        baseline = [line for line in lines if line.timestamp < recent_start]
        recent = [line for line in lines if line.timestamp >= recent_start]
        return await self._patterns.evaluate(baseline, recent)
