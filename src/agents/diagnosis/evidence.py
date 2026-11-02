"""Agent 2 local evidence handoff; kept here so shared contracts stay unchanged."""

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, JsonValue

from src.contracts.base import Contract
from src.contracts.tasks import AgentTask


class ResolvedEvidence(Contract):
    """Bounded evidence content resolved for one incident before it reaches Agent 2."""

    evidence_id: Annotated[str, Field(min_length=1)]
    incident_id: Annotated[str, Field(min_length=1)]
    evidence_type: Literal["metric", "log", "k8s_event", "git_change", "config", "trace"]
    source: Annotated[str, Field(min_length=1)]
    observed_at: AwareDatetime
    summary: Annotated[str, Field(min_length=1)]
    value: dict[str, JsonValue]


def validate_resolved_evidence_scope(
    task: AgentTask, resolved_evidence: Sequence[ResolvedEvidence]
) -> None:
    """Reject evidence that the task did not request or that belongs to another incident."""
    requested_refs = task.task_context.evidence_refs
    if len(requested_refs) != len(set(requested_refs)):
        raise ValueError("Evidence references must be unique within a task")

    resolved_ids = [item.evidence_id for item in resolved_evidence]
    if len(resolved_ids) != len(set(resolved_ids)):
        raise ValueError("Resolved evidence identifiers must be unique within a task")
    if not set(resolved_ids) <= set(requested_refs):
        raise ValueError("Resolved evidence must have been requested by the task")
    if any(item.incident_id != task.incident_id for item in resolved_evidence):
        raise ValueError("Resolved evidence must belong to the task incident")
