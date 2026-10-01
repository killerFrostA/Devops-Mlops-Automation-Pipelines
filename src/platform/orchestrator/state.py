"""Validated, compact checkpoints for the analysis supervisor."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from src.contracts.base import Contract
from src.contracts.enums import AgentId, Environment, Priority, ResultStatus, Severity
from src.contracts.tasks import AgentResult, AgentTask
from src.platform.orchestrator.decision.contracts import DecisionDisposition, DecisionPlan


class WorkflowStage(StrEnum):
    INGESTED = "INGESTED"
    DIAGNOSING = "DIAGNOSING"
    ASSESSING = "ASSESSING"
    WAITING_FOR_EVENT = "WAITING_FOR_EVENT"
    WAITING_FOR_EVIDENCE = "WAITING_FOR_EVIDENCE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    CLOSED = "CLOSED"
    BLOCKED = "BLOCKED"


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    DISPATCHING = "DISPATCHING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    TIMED_OUT = "TIMED_OUT"


TERMINAL_TASK_STATUSES = frozenset(
    {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.NOT_IMPLEMENTED, TaskStatus.TIMED_OUT}
)


class EventReceipt(Contract):
    message_id: Annotated[str, Field(min_length=1)]
    idempotency_key: Annotated[str, Field(min_length=1)]
    fingerprint_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class TaskRecord(Contract):
    task: AgentTask
    status: TaskStatus = TaskStatus.PENDING
    result: AgentResult | None = None
    error_code: str | None = None

    @model_validator(mode="after")
    def check_result(self) -> "TaskRecord":
        expected = {
            ResultStatus.SUCCEEDED: TaskStatus.SUCCEEDED,
            ResultStatus.FAILED: TaskStatus.FAILED,
            ResultStatus.NOT_IMPLEMENTED: TaskStatus.NOT_IMPLEMENTED,
        }
        if self.result is not None:
            if (
                self.result.task_id != self.task.task_id
                or self.result.incident_id != self.task.incident_id
                or self.result.agent_id != self.task.agent_id
                or self.status != expected[self.result.status]
            ):
                raise ValueError("Task result does not match the dispatched task and status")
            if self.error_code is not None:
                raise ValueError("A result and an error code cannot coexist")
        elif self.status in {TaskStatus.SUCCEEDED, TaskStatus.NOT_IMPLEMENTED}:
            raise ValueError("Completed task status requires a result")
        elif self.status in {TaskStatus.FAILED, TaskStatus.TIMED_OUT}:
            if not self.error_code:
                raise ValueError("Failed task status requires an error code")
        elif self.error_code is not None:
            raise ValueError("Unfinished task cannot have an error code")
        return self


class WorkflowState(Contract):
    """One incident checkpoint; no raw telemetry, secrets or mutable action intent."""

    schema_version: Literal["1"] = "1"
    incident_id: Annotated[str, Field(min_length=1)]
    revision: Annotated[int, Field(ge=1)] = 1
    stage: WorkflowStage
    environment: Environment
    service_id: Annotated[str, Field(min_length=1)]
    namespace: Annotated[str, Field(min_length=1)]
    service_version: Annotated[str, Field(min_length=1)]
    priority: Priority
    severity: Severity
    trace_id: Annotated[str, Field(min_length=1)]
    receipts: tuple[EventReceipt, ...] = ()
    decisions: tuple[DecisionPlan, ...] = ()
    tasks: tuple[TaskRecord, ...] = ()
    selected_action_id: str | None = None
    approval_id: str | None = None
    updated_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def check_consistency(self) -> "WorkflowState":
        if len({item.task.task_id for item in self.tasks}) != len(self.tasks):
            raise ValueError("Duplicate task IDs in workflow checkpoint")
        result_ids = [item.result.result_id for item in self.tasks if item.result is not None]
        if len(set(result_ids)) != len(result_ids):
            raise ValueError("Duplicate result IDs in workflow checkpoint")
        if len({item.message_id for item in self.receipts}) != len(self.receipts):
            raise ValueError("Duplicate event message IDs in workflow checkpoint")
        if len({item.idempotency_key for item in self.receipts}) != len(self.receipts):
            raise ValueError("Duplicate event idempotency keys in workflow checkpoint")
        if len({decision.event_id for decision in self.decisions}) != len(self.decisions):
            raise ValueError("Duplicate decision event IDs in workflow checkpoint")
        receipt_ids = {receipt.message_id for receipt in self.receipts}
        if any(
            decision.incident_id != self.incident_id or decision.event_id not in receipt_ids
            for decision in self.decisions
        ):
            raise ValueError("Decision is not bound to this incident and event")
        for item in self.tasks:
            if item.task.incident_id != self.incident_id:
                raise ValueError("Task incident differs from workflow incident")
            context = item.task.task_context
            if (
                context.environment != self.environment
                or context.service_id != self.service_id
                or context.namespace != self.namespace
            ):
                raise ValueError("Task scope differs from workflow scope")
        if self.stage == WorkflowStage.WAITING_FOR_EVENT and any(
            item.status != TaskStatus.SUCCEEDED for item in self.tasks
        ):
            raise ValueError("Workflow cannot await the next event with unfinished tasks")
        blocked_by_task = any(
            item.status in TERMINAL_TASK_STATUSES - {TaskStatus.SUCCEEDED} for item in self.tasks
        )
        blocked_by_decision = bool(
            self.decisions and self.decisions[-1].disposition == DecisionDisposition.BLOCKED
        )
        if self.stage == WorkflowStage.BLOCKED and not (blocked_by_task or blocked_by_decision):
            raise ValueError("Blocked workflow requires a failed task or blocking decision")
        return self

    @property
    def active_agents(self) -> tuple[AgentId, ...]:
        return tuple(
            item.task.agent_id
            for item in self.tasks
            if item.status in {TaskStatus.PENDING, TaskStatus.DISPATCHING}
        )
