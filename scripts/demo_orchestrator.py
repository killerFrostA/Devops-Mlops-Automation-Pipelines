"""Run recommendation routing with synthetic evidence and mock specialists."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from src.contracts.context import EvidenceReference, IncidentContext
from src.contracts.enums import AgentId, Environment, ResultStatus
from src.contracts.events import EventEnvelope
from src.contracts.tasks import AgentResult, AgentTask
from src.platform.orchestrator.repository import SQLiteWorkflowRepository
from src.platform.orchestrator.service import Orchestrator

ROOT = Path(__file__).resolve().parents[1]


class DemoContextReader:
    async def get_incident_context(self, incident_id: str) -> IncidentContext:
        return IncidentContext(
            incident_id=incident_id,
            version=1,
            environment=Environment.TEST,
            status="OPEN",
            evidence_refs=(
                EvidenceReference(
                    evidence_id="EVD-synthetic-001",
                    incident_id=incident_id,
                    source="demo",
                    object_uri="memory://demo",
                    checksum_sha256="a" * 64,
                    observed_at=datetime.now(UTC),
                ),
            ),
        )


class DemoAnalysisAgent:
    def __init__(self, agent_id: AgentId) -> None:
        self.agent_id = agent_id

    async def handle(self, task: AgentTask) -> AgentResult:
        return AgentResult(
            result_id=str(uuid4()),
            task_id=task.task_id,
            incident_id=task.incident_id,
            agent_id=self.agent_id,
            status=ResultStatus.SUCCEEDED,
            rationale_summary="Synthetic analysis completed",
        )


def fixture_event(topic: str) -> EventEnvelope:
    fixture = ROOT / "contracts/examples/v1" / f"{topic}.json"
    document = json.loads(fixture.read_text(encoding="utf-8"))
    document.update(
        message_id=str(uuid4()),
        idempotency_key=str(uuid4()),
        timestamp=datetime.now(UTC).isoformat(),
    )
    return EventEnvelope.model_validate(document)


async def main() -> None:
    with TemporaryDirectory(prefix="orchestrator-demo-") as temporary:
        repository = SQLiteWorkflowRepository(Path(temporary) / "checkpoints.sqlite3")
        supervisor = Orchestrator(
            repository,
            DemoContextReader(),
            {
                AgentId.DIAGNOSIS: DemoAnalysisAgent(AgentId.DIAGNOSIS),
                AgentId.SECURITY_QUALITY: DemoAnalysisAgent(AgentId.SECURITY_QUALITY),
            },
        )
        await supervisor.run_event(fixture_event("monitoring.anomaly.detected"))
        state = await supervisor.run_event(fixture_event("diagnosis.completed"))
        restarted = SQLiteWorkflowRepository(repository.path).get(state.incident_id)
        print(
            json.dumps(
                {
                    "incident_id": state.incident_id,
                    "stage": state.stage,
                    "decisions": [
                        {
                            "event_topic": decision.topic,
                            "disposition": decision.disposition,
                            "next_agents": decision.next_agents,
                            "candidate_action_ids": [
                                candidate.action_id for candidate in decision.candidates
                            ],
                        }
                        for decision in state.decisions
                    ],
                    "task_statuses": [item.status for item in state.tasks],
                    "checkpoint_survived_restart": restarted == state,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
