"""Targeted, deterministic evidence collection before the diagnosis rules run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import ValidationError

from src.agents.diagnosis.contracts import (
    DiagnosisInput,
    EvidenceItem,
    RecentChangeEvent,
    ToolCallRecord,
)
from src.agents.diagnosis.tools import Toolbox, ToolContext

# Evidence names used in recommendations and replan requests, mapped to tool operations.
REQUIRED_EVIDENCE_OPERATIONS: dict[str, tuple[str, str]] = {
    "memory_limit": ("prometheus", "memory_percent"),
    "memory_trend": ("prometheus", "memory_percent"),
    "heap_profile": ("prometheus", "heap_usage"),
    "heap_usage_breakdown": ("prometheus", "heap_usage"),
    "gc_metrics": ("prometheus", "gc_pause"),
    "cpu_percent": ("prometheus", "cpu_percent"),
    "cpu_limit": ("prometheus", "cpu_percent"),
    "latency_ms": ("prometheus", "latency_p95_ms"),
    "connection_metrics": ("prometheus", "error_rate"),
    "deployment_timeline": ("kubernetes", "deployment_history"),
    "recent_change_events": ("kubernetes", "deployment_history"),
    "previous_stable_version": ("kubernetes", "deployment_history"),
    "oom_events": ("kubernetes", "events"),
    "restart_history": ("kubernetes", "pod_status"),
    "exit_code": ("kubernetes", "pod_status"),
    "recent_logs": ("loki", "error_logs"),
    "previous_container_logs": ("loki", "error_logs"),
    "dependency_health": ("loki", "error_logs"),
    "config_diff": ("git", "recent_changes"),
}

# Always useful: pod state, rollouts, error logs and recent changes.
_COMMON_PLAN = (
    ("kubernetes", "pod_status"),
    ("kubernetes", "events"),
    ("kubernetes", "deployment_history"),
    ("loki", "error_logs"),
    ("git", "recent_changes"),
)
_CATEGORY_PLAN = {
    "memory": (("prometheus", "memory_percent"), ("prometheus", "heap_usage")),
    "cpu": (("prometheus", "cpu_percent"), ("prometheus", "request_rate")),
    "errors": (
        ("prometheus", "error_rate"),
        ("prometheus", "latency_p95_ms"),
        ("prometheus", "request_rate"),
    ),
}


@dataclass(frozen=True)
class Investigation:
    evidence: tuple[EvidenceItem, ...]
    change_events: tuple[RecentChangeEvent, ...]
    tool_calls: tuple[ToolCallRecord, ...]
    limitations: tuple[str, ...]


def _focus(payload: DiagnosisInput) -> set[str]:
    signals = payload.signals
    context = signals.get("agent1_context")
    categories = context.get("categories", []) if isinstance(context, dict) else []
    text = " ".join([*(c for c in categories if isinstance(c, str)), *signals]).casefold()
    focus = set()
    if "memory" in text or "oom" in text:
        focus.add("memory")
    if "cpu" in text:
        focus.add("cpu")
    if any(word in text for word in ("error", "latency", "burn")):
        focus.add("errors")
    return focus or {"memory", "cpu", "errors"}


def plan_investigation(payload: DiagnosisInput) -> tuple[list[tuple[str, str]], list[str]]:
    """Return the ordered tool calls and the requested evidence that no tool can provide."""
    plan = list(_COMMON_PLAN)
    for focus in sorted(_focus(payload)):
        plan.extend(_CATEGORY_PLAN[focus])
    unknown = []
    for name in payload.required_evidence:
        operation = REQUIRED_EVIDENCE_OPERATIONS.get(name)
        if operation is None:
            unknown.append(name)
        else:
            plan.append(operation)
    return list(dict.fromkeys(plan)), unknown


def change_events_from_evidence(
    evidence: tuple[EvidenceItem, ...],
) -> tuple[RecentChangeEvent, ...]:
    """Dated deployments/commits collected by tools become temporal RCA change events."""
    events = []
    for item in evidence:
        if not {"change_id", "description", "changed_at"} <= item.value.keys():
            continue
        try:
            events.append(
                RecentChangeEvent(
                    change_id=item.value["change_id"],
                    description=item.value["description"],
                    changed_at=item.value["changed_at"],
                )
            )
        except ValidationError:
            continue
    return tuple(events)


def tool_context(payload: DiagnosisInput, deadline: datetime) -> ToolContext:
    return ToolContext(
        incident_id=payload.incident_id,
        service_id=payload.service_id,
        namespace=payload.namespace,
        deadline=deadline,
    )


def investigate(payload: DiagnosisInput, toolbox: Toolbox, deadline: datetime) -> Investigation:
    plan, unknown = plan_investigation(payload)
    context = tool_context(payload, deadline)
    evidence: list[EvidenceItem] = []
    records: list[ToolCallRecord] = []
    limitations = [
        f"Requested evidence '{name}' has no matching read-only tool and was not collected."
        for name in unknown
    ]
    for tool, operation in plan:
        if not toolbox.available(tool, operation):
            continue
        result = toolbox.call(tool, operation, context)
        records.append(result.record)
        evidence.extend(result.evidence)
        if result.record.status != "succeeded":
            limitations.append(
                f"Tool {tool}.{operation} {result.record.status}: {result.record.error}."
            )
    for name in payload.required_evidence:
        required = REQUIRED_EVIDENCE_OPERATIONS.get(name)
        if required is not None and not toolbox.available(*required):
            limitations.append(
                f"Requested evidence '{name}' needs {required[0]}.{required[1]}, "
                "which is not configured."
            )
    collected = tuple(evidence)
    return Investigation(
        evidence=collected,
        change_events=change_events_from_evidence(collected),
        tool_calls=tuple(records),
        limitations=tuple(dict.fromkeys(limitations)),
    )
