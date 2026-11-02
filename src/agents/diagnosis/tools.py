"""Read-only evidence tools used by Agent 2 to collect targeted incident evidence.

Every tool exposes a finite set of operations (section 18.1 of the architecture: tool
allowlisting). No operation writes to any system: Prometheus and Loki are queried over HTTP,
Kubernetes only through ``kubectl get``, Git only through ``git log``. Commands are built
from fixed argument lists, never from a shell string, and every call is bounded by the task
deadline. ``StaticEvidenceTool`` replays recorded responses for offline scenarios and tests.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from pydantic import ValidationError

from src.agents.diagnosis.contracts import EvidenceItem, ToolCallRecord
from src.agents.diagnosis.signatures import line_signatures


DEFAULT_TOOL_TIMEOUT_SECONDS = 10.0
DEFAULT_LOOKBACK = timedelta(hours=1)
MAX_LOG_LINES = 5
MAX_LOG_LINE_LENGTH = 300
# Kubernetes names and label values; anything else is refused before a query is built.
_SAFE_NAME = re.compile(r"^[a-z0-9]([a-z0-9.\-]{0,251}[a-z0-9])?$")
_DEPENDENCY_HOST = re.compile(
    r"(?:to|connect(?:ing)? to|host|upstream|tcp)\s+['\"]?([a-z0-9][a-z0-9.\-]*[a-z0-9])(?::\d+)?",
    re.IGNORECASE,
)


_DEPENDENCY_SIGNATURES = frozenset({"connection_refused", "dependency_timeout", "pool_exhausted"})


class ToolError(RuntimeError):
    """A tool could not produce evidence; the diagnosis continues without it."""


@dataclass(frozen=True)
class ToolContext:
    incident_id: str
    service_id: str
    namespace: str
    deadline: datetime
    lookback: timedelta = DEFAULT_LOOKBACK

    def remaining_seconds(self, cap: float) -> float:
        remaining = (self.deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise ToolError("task deadline reached before the tool call")
        return min(cap, remaining)


class EvidenceTool(Protocol):
    name: str
    description: str
    operations: Mapping[str, str]

    def run(self, operation: str, context: ToolContext) -> list[EvidenceItem]: ...


def _safe_name(value: str, field_name: str) -> str:
    if not _SAFE_NAME.match(value):
        raise ToolError(f"{field_name} '{value}' is not a valid Kubernetes name")
    return value


def _evidence_id(context: ToolContext, tool: str, operation: str, index: int) -> str:
    return f"EVD-{context.incident_id}-{tool.upper()}-{operation.upper()}-{index:02d}"


def _http_get_json(base_url: str, path: str, params: dict[str, str], timeout: float) -> Any:
    url = f"{base_url.rstrip('/')}{path}?{urlencode(params)}"
    try:
        with urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed http(s) base URL
            return json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise ToolError(f"GET {path} failed: {exc}") from exc


def _run_readonly(argv: Sequence[str], timeout: float) -> str:
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            list(argv), capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolError(f"{argv[0]} failed: {exc}") from exc
    if completed.returncode != 0:
        raise ToolError(f"{argv[0]} exited with {completed.returncode}: {completed.stderr.strip()[:300]}")
    return completed.stdout


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.utcoffset() is not None else None


class PrometheusMetricsTool:
    """Instant PromQL queries for the metrics the diagnosis rules understand."""

    name = "prometheus"
    description = "Read current service metrics from Prometheus (read-only PromQL queries)."
    operations = {
        "memory_percent": "Container memory working set as a percentage of its limit.",
        "cpu_percent": "Container CPU usage as a percentage of its limit.",
        "restart_count": "Container restarts during the lookback window.",
        "error_rate": "Share of HTTP 5xx responses, in percent.",
        "latency_p95_ms": "95th percentile request latency in milliseconds.",
        "request_rate": "Requests per second now and one day earlier.",
        "heap_usage": "Process heap/resident memory in megabytes.",
        "gc_pause": "Garbage-collection pause time per second.",
    }
    _QUERIES = {
        "memory_percent": (
            "100 * max(container_memory_working_set_bytes{{{pods}}}) / "
            "max(kube_pod_container_resource_limits{{{pods},resource=\"memory\"}})"
        ),
        "cpu_percent": (
            "100 * sum(rate(container_cpu_usage_seconds_total{{{pods}}}[5m])) / "
            "sum(kube_pod_container_resource_limits{{{pods},resource=\"cpu\"}})"
        ),
        "restart_count": "max(increase(kube_pod_container_status_restarts_total{{{pods}}}[{window}]))",
        "error_rate": (
            "100 * sum(rate(http_requests_total{{{service},status=~\"5..\"}}[5m])) / "
            "sum(rate(http_requests_total{{{service}}}[5m]))"
        ),
        "latency_p95_ms": (
            "1000 * histogram_quantile(0.95, sum by (le) "
            "(rate(http_request_duration_seconds_bucket{{{service}}}[5m])))"
        ),
        "request_rate": "sum(rate(http_requests_total{{{service}}}[5m]))",
        "heap_usage": "max(process_resident_memory_bytes{{{pods}}}) / 1048576",
        "gc_pause": "sum(rate(go_gc_duration_seconds_sum{{{pods}}}[5m]))",
    }
    _VALUE_KEYS = {
        "memory_percent": "memory_percent",
        "cpu_percent": "cpu_percent",
        "restart_count": "restart_count",
        "error_rate": "error_rate_percent",
        "latency_p95_ms": "latency_ms",
        "request_rate": "rps",
        "heap_usage": "heap_mb",
        "gc_pause": "gc_pause_seconds_per_second",
    }

    def __init__(self, base_url: str, timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS) -> None:
        self._base_url = base_url
        self._timeout = timeout

    def _query(self, promql: str, context: ToolContext) -> float | None:
        data = _http_get_json(
            self._base_url,
            "/api/v1/query",
            {"query": promql},
            context.remaining_seconds(self._timeout),
        )
        if data.get("status") != "success":
            raise ToolError(f"Prometheus returned status {data.get('status')!r}")
        values = []
        for sample in data.get("data", {}).get("result", []):
            try:
                values.append(float(sample["value"][1]))
            except (KeyError, IndexError, TypeError, ValueError):
                continue
        finite = [value for value in values if value == value and abs(value) != float("inf")]
        return max(finite) if finite else None

    def run(self, operation: str, context: ToolContext) -> list[EvidenceItem]:
        service = _safe_name(context.service_id, "service")
        namespace = _safe_name(context.namespace, "namespace")
        selectors = {
            "pods": f'namespace="{namespace}",pod=~"{service}-.*"',
            "service": f'namespace="{namespace}",service="{service}"',
            "window": f"{int(context.lookback.total_seconds())}s",
        }
        promql = self._QUERIES[operation].format(**selectors)
        value = self._query(promql, context)
        if value is None:
            return []
        key = self._VALUE_KEYS[operation]
        payload: dict[str, Any] = {key: round(value, 3), "query": promql}
        summary = f"Prometheus reports {key}={value:.3g} for {service}."
        if operation == "request_rate":
            baseline = self._query(f"{promql} offset 1d", context)
            if baseline is not None:
                payload["rps_baseline"] = round(baseline, 3)
                summary = (
                    f"Prometheus reports {value:.3g} requests/s for {service} "
                    f"versus {baseline:.3g} one day earlier."
                )
        return [
            EvidenceItem(
                evidence_id=_evidence_id(context, self.name, operation, 1),
                evidence_type="metric",
                source="prometheus",
                provenance="provided",
                observed_at=datetime.now(UTC),
                summary=summary,
                value=payload,
            )
        ]


class LokiLogsTool:
    """Error-level log lines for the service, reduced to signatures and samples."""

    name = "loki"
    description = "Read recent error log lines of the service from Loki (read-only LogQL)."
    operations = {"error_logs": "Recent error, exception, timeout and OOM log lines."}

    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
        app_label: str = "app",
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._app_label = app_label

    def run(self, operation: str, context: ToolContext) -> list[EvidenceItem]:
        service = _safe_name(context.service_id, "service")
        namespace = _safe_name(context.namespace, "namespace")
        logql = (
            f'{{namespace="{namespace}",{self._app_label}="{service}"}} '
            '|~ "(?i)(error|exception|fatal|refused|timed? ?out|oom|invalid|exhausted)"'
        )
        end = datetime.now(UTC)
        data = _http_get_json(
            self._base_url,
            "/loki/api/v1/query_range",
            {
                "query": logql,
                "start": str(int((end - context.lookback).timestamp() * 1e9)),
                "end": str(int(end.timestamp() * 1e9)),
                "limit": "200",
                "direction": "backward",
            },
            context.remaining_seconds(self._timeout),
        )
        lines = [
            entry[1]
            for stream in data.get("data", {}).get("result", [])
            for entry in stream.get("values", [])
            if isinstance(entry, list) and len(entry) == 2 and isinstance(entry[1], str)
        ]
        if not lines:
            return []
        return [_log_evidence(context, self.name, lines, source="loki")]


def _log_evidence(
    context: ToolContext,
    tool: str,
    lines: Sequence[str],
    *,
    source: str,
) -> EvidenceItem:
    samples = [line[:MAX_LOG_LINE_LENGTH] for line, _ in Counter(lines).most_common(MAX_LOG_LINES)]
    signatures = sorted({name for line in lines for name in line_signatures(line)})
    value: dict[str, Any] = {"line_count": len(lines), "lines": samples, "signatures": signatures}
    for line in lines:
        match = _DEPENDENCY_HOST.search(line)
        if match and _DEPENDENCY_SIGNATURES & line_signatures(line):
            value["dependency"] = match.group(1).split(".")[0]
            break
    described = ", ".join(signatures) if signatures else "no known signature"
    return EvidenceItem(
        evidence_id=_evidence_id(context, tool, "error_logs", 1),
        evidence_type="log",
        source=source,
        provenance="provided",
        observed_at=datetime.now(UTC),
        summary=f"{len(lines)} error log lines for {context.service_id} ({described}).",
        value=value,
    )


class KubernetesTool:
    """Pod state, events and rollout history through read-only ``kubectl get`` calls."""

    name = "kubernetes"
    description = "Read pod status, events and deployment history from Kubernetes (kubectl get)."
    operations = {
        "pod_status": "Restart counts, waiting reasons and last termination of the pods.",
        "events": "Recent warning events of the service pods.",
        "deployment_history": "Dated rollout revisions of the service deployment.",
    }

    def __init__(
        self,
        kubectl: str = "kubectl",
        context_name: str | None = None,
        app_label: str = "app",
        timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
        runner: Callable[[Sequence[str], float], str] = _run_readonly,
    ) -> None:
        self._kubectl = kubectl
        self._context_name = context_name
        self._app_label = app_label
        self._timeout = timeout
        self._runner = runner

    def _get(self, resource: str, context: ToolContext, *extra: str) -> dict[str, Any]:
        argv = [self._kubectl]
        if self._context_name:
            argv += ["--context", self._context_name]
        argv += ["get", resource, "-n", _safe_name(context.namespace, "namespace"), *extra, "-o", "json"]
        output = self._runner(argv, context.remaining_seconds(self._timeout))
        try:
            return json.loads(output)
        except json.JSONDecodeError as exc:
            raise ToolError(f"kubectl get {resource} returned invalid JSON") from exc

    def _selector(self, context: ToolContext) -> str:
        return f"{self._app_label}={_safe_name(context.service_id, 'service')}"

    def run(self, operation: str, context: ToolContext) -> list[EvidenceItem]:
        if operation == "pod_status":
            return self._pod_status(context)
        if operation == "events":
            return self._events(context)
        return self._deployment_history(context)

    def _pod_status(self, context: ToolContext) -> list[EvidenceItem]:
        pods = self._get("pods", context, "-l", self._selector(context)).get("items", [])
        restarts, waiting, terminated, exit_codes = 0, Counter(), Counter(), []
        for pod in pods:
            for status in pod.get("status", {}).get("containerStatuses", []) or []:
                restarts = max(restarts, int(status.get("restartCount", 0) or 0))
                reason = (status.get("state", {}).get("waiting") or {}).get("reason")
                if reason:
                    waiting[reason] += 1
                last = (status.get("lastState", {}).get("terminated") or {})
                if last.get("reason"):
                    terminated[last["reason"]] += 1
                if isinstance(last.get("exitCode"), int):
                    exit_codes.append(last["exitCode"])
        if not pods:
            return []
        reasons = [*waiting, *terminated]
        return [
            EvidenceItem(
                evidence_id=_evidence_id(context, self.name, "pod_status", 1),
                evidence_type="k8s_event",
                source="kubernetes",
                provenance="provided",
                observed_at=datetime.now(UTC),
                summary=(
                    f"{len(pods)} pods of {context.service_id}: max {restarts} restarts"
                    + (f"; reasons {', '.join(reasons)}" if reasons else "")
                    + "."
                ),
                value={
                    "restart_count": restarts,
                    "reason": " ".join(reasons),
                    "waiting_reasons": dict(waiting),
                    "termination_reasons": dict(terminated),
                    "exit_codes": exit_codes,
                },
            )
        ]

    def _events(self, context: ToolContext) -> list[EvidenceItem]:
        items = self._get("events", context, "--field-selector", "type=Warning").get("items", [])
        prefix = f"{context.service_id}-"
        relevant = [
            event
            for event in items
            if str(event.get("involvedObject", {}).get("name", "")).startswith(prefix)
        ]
        if not relevant:
            return []
        reasons = Counter(str(event.get("reason", "")) for event in relevant)
        messages = [str(event.get("message", ""))[:MAX_LOG_LINE_LENGTH] for event in relevant]
        return [
            EvidenceItem(
                evidence_id=_evidence_id(context, self.name, "events", 1),
                evidence_type="k8s_event",
                source="kubernetes",
                provenance="provided",
                observed_at=datetime.now(UTC),
                summary=f"Kubernetes warning events: {dict(reasons)}.",
                value={
                    "reason": " ".join(reasons),
                    "event_counts": dict(reasons),
                    "lines": messages[:MAX_LOG_LINES],
                },
            )
        ]

    def _deployment_history(self, context: ToolContext) -> list[EvidenceItem]:
        replica_sets = self._get("replicasets", context, "-l", self._selector(context)).get("items", [])
        cutoff = datetime.now(UTC) - context.lookback * 24
        evidence = []
        for replica_set in replica_sets:
            metadata = replica_set.get("metadata", {})
            created = _parse_time(metadata.get("creationTimestamp"))
            revision = metadata.get("annotations", {}).get("deployment.kubernetes.io/revision")
            if created is None or revision is None or created < cutoff:
                continue
            containers = replica_set.get("spec", {}).get("template", {}).get("spec", {}).get(
                "containers", []
            )
            images = ", ".join(str(container.get("image")) for container in containers)
            evidence.append((created, str(revision), images))
        items = []
        for index, (created, revision, images) in enumerate(sorted(evidence), start=1):
            change_id = f"{context.service_id}-rev{revision}"
            items.append(
                EvidenceItem(
                    evidence_id=_evidence_id(context, self.name, "deployment_history", index),
                    evidence_type="k8s_event",
                    source="kubernetes",
                    provenance="provided",
                    observed_at=created,
                    summary=f"Deployment revision {revision} rolled out at {created.isoformat()} ({images}).",
                    value={
                        "change_id": change_id,
                        "description": f"Deployment revision {revision} ({images})",
                        "changed_at": created.isoformat(),
                    },
                )
            )
        return items


class GitChangesTool:
    """Recent commits of the service repository through read-only ``git log``."""

    name = "git"
    description = "Read recent commits and changed files of the service repository (git log)."
    operations = {"recent_changes": "Commits in the lookback window with their changed files."}

    def __init__(
        self,
        repository: Path,
        git: str = "git",
        timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
        runner: Callable[[Sequence[str], float], str] = _run_readonly,
        max_commits: int = 10,
    ) -> None:
        self._repository = repository
        self._git = git
        self._timeout = timeout
        self._runner = runner
        self._max_commits = max_commits

    def run(self, operation: str, context: ToolContext) -> list[EvidenceItem]:
        since = (datetime.now(UTC) - context.lookback * 24).isoformat()
        output = self._runner(
            [
                self._git, "-C", str(self._repository), "log", f"--since={since}",
                f"--max-count={self._max_commits}", "--name-only",
                "--format=%x1e%H%x1f%cI%x1f%s",
            ],
            context.remaining_seconds(self._timeout),
        )
        items = []
        for index, record in enumerate((part for part in output.split("\x1e") if part.strip()), start=1):
            header, _, files_block = record.partition("\n")
            sha, committed_at, subject = (header.split("\x1f") + ["", "", ""])[:3]
            changed_at = _parse_time(committed_at)
            if changed_at is None:
                continue
            files = [line.strip() for line in files_block.splitlines() if line.strip()]
            description = f"{subject} [{', '.join(files[:10])}]" if files else subject
            items.append(
                EvidenceItem(
                    evidence_id=_evidence_id(context, self.name, "recent_changes", index),
                    evidence_type="git_change",
                    source="git",
                    provenance="provided",
                    observed_at=changed_at,
                    summary=f"Commit {sha[:12]} at {changed_at.isoformat()}: {description}.",
                    value={
                        "change_id": sha[:12],
                        "description": description,
                        "changed_at": changed_at.isoformat(),
                        "files": files,
                    },
                )
            )
        return items


class StaticEvidenceTool:
    """Replays recorded evidence per operation; used for offline scenarios and tests."""

    def __init__(
        self,
        name: str,
        responses: Mapping[str, Sequence[Mapping[str, Any]]],
        description: str | None = None,
    ) -> None:
        self.name = name
        self.description = description or f"Recorded {name} evidence (offline replay)."
        self.operations = {operation: f"Recorded {operation} evidence." for operation in responses}
        self._responses = responses

    def run(self, operation: str, context: ToolContext) -> list[EvidenceItem]:
        items = []
        for index, raw in enumerate(self._responses.get(operation, ()), start=1):
            record = {
                "evidence_id": _evidence_id(context, self.name, operation, index),
                "source": self.name,
                "provenance": "fixture",
                **raw,
            }
            try:
                items.append(EvidenceItem.model_validate(record))
            except ValidationError as exc:
                raise ToolError(f"Invalid recorded evidence for {self.name}.{operation}: {exc}") from exc
        return items


@dataclass
class ToolCallResult:
    record: ToolCallRecord
    evidence: tuple[EvidenceItem, ...] = ()


@dataclass
class Toolbox:
    """Allowlisted, deadline-bounded access to evidence tools with a simple circuit breaker."""

    tools: Sequence[EvidenceTool]
    max_failures_per_tool: int = 2
    max_calls: int = 25
    _failures: Counter[str] = field(default_factory=Counter)
    _cache: dict[tuple[str, str, str], ToolCallResult] = field(default_factory=dict)
    _calls: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("Evidence tool names must be unique")
        self._by_name = {tool.name: tool for tool in self.tools}

    def available(self, tool: str, operation: str) -> bool:
        return tool in self._by_name and operation in self._by_name[tool].operations

    def definitions(self) -> list[dict[str, Any]]:
        """Tool definitions for the Claude Messages API (strict JSON schemas)."""
        return [
            {
                "name": tool.name,
                "description": tool.description
                + " Operations: "
                + "; ".join(f"{name}: {text}" for name, text in tool.operations.items()),
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": sorted(tool.operations)}
                    },
                    "required": ["operation"],
                    "additionalProperties": False,
                },
            }
            for tool in self.tools
        ]

    def call(
        self,
        tool_name: str,
        operation: str,
        context: ToolContext,
        requested_by: str = "investigation_plan",
    ) -> ToolCallResult:
        def record(status: str, refs: tuple[str, ...] = (), error: str | None = None) -> ToolCallRecord:
            return ToolCallRecord(
                tool=tool_name,
                operation=operation,
                requested_by=requested_by,
                status=status,
                evidence_refs=refs,
                error=error,
            )

        if not self.available(tool_name, operation):
            return ToolCallResult(record("skipped", error="tool or operation is not allowlisted"))
        key = (context.incident_id, tool_name, operation)
        if key in self._cache:
            cached = self._cache[key]
            return ToolCallResult(cached.record.model_copy(update={"requested_by": requested_by}), cached.evidence)
        if self._failures[tool_name] >= self.max_failures_per_tool:
            return ToolCallResult(record("skipped", error="circuit open after repeated failures"))
        if self._calls[context.incident_id] >= self.max_calls:
            return ToolCallResult(record("skipped", error="tool call budget exhausted"))
        if context.deadline <= datetime.now(UTC):
            return ToolCallResult(record("skipped", error="task deadline reached"))

        self._calls[context.incident_id] += 1
        try:
            evidence = tuple(self._by_name[tool_name].run(operation, context))
        except Exception as exc:  # A failing tool must never stop the diagnosis.
            self._failures[tool_name] += 1
            return ToolCallResult(record("failed", error=f"{type(exc).__name__}: {exc}"[:300]))
        result = ToolCallResult(
            record("succeeded", tuple(item.evidence_id for item in evidence)), evidence
        )
        self._cache[key] = result
        return result


def build_toolbox_from_env(env: Mapping[str, str] | None = None) -> Toolbox | None:
    """Create the live toolbox from PIDS_* settings; None when no tool is configured."""
    env = os.environ if env is None else env
    app_label = env.get("PIDS_DIAGNOSIS_K8S_APP_LABEL", "app")
    tools: list[EvidenceTool] = []
    if env.get("PIDS_PROMETHEUS_URL"):
        tools.append(PrometheusMetricsTool(env["PIDS_PROMETHEUS_URL"]))
    if env.get("PIDS_LOKI_URL"):
        tools.append(LokiLogsTool(env["PIDS_LOKI_URL"], app_label=app_label))
    if env.get("PIDS_DIAGNOSIS_KUBECTL"):
        tools.append(
            KubernetesTool(
                kubectl=env["PIDS_DIAGNOSIS_KUBECTL"],
                context_name=env.get("PIDS_DIAGNOSIS_KUBE_CONTEXT") or None,
                app_label=app_label,
            )
        )
    if env.get("PIDS_DIAGNOSIS_GIT_REPO"):
        tools.append(GitChangesTool(Path(env["PIDS_DIAGNOSIS_GIT_REPO"])))
    return Toolbox(tools) if tools else None


def load_static_toolbox(path: Path) -> Toolbox:
    """Load ``{"tools": {"<tool>": {"<operation>": [evidence, ...]}}}`` recorded responses."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolError(f"Cannot read recorded tool responses '{path}': {exc}") from exc
    return static_toolbox(document.get("tools", {}))


def static_toolbox(tools: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> Toolbox:
    if not isinstance(tools, Mapping):
        raise ToolError("Recorded tool responses must be an object keyed by tool name")
    return Toolbox([StaticEvidenceTool(name, responses) for name, responses in tools.items()])
