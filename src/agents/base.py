from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from src.contracts.enums import AgentId, ResultStatus
from src.contracts.tasks import AgentResult, AgentTask


class Agent(Protocol):
    agent_id: AgentId

    async def handle(self, task: AgentTask) -> AgentResult: ...


class SkeletonAgent:
    """A replaceable implementation slot that reports its incomplete status honestly."""

    agent_id: AgentId
    responsibility: str

    async def handle(self, task: AgentTask) -> AgentResult:
        if task.agent_id != self.agent_id:
            raise ValueError("Task addressed to a different agent")
        if task.deadline <= datetime.now(UTC):
            raise ValueError("Task deadline has expired")
        return AgentResult(
            result_id=str(uuid4()),
            task_id=task.task_id,
            incident_id=task.incident_id,
            agent_id=self.agent_id,
            status=ResultStatus.NOT_IMPLEMENTED,
            rationale_summary=f"Implementation pending: {self.responsibility}",
        )
