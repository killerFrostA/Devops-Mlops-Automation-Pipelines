"""Behavioral tests for the local orchestrator, including failure and replay paths."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.contracts.context import EvidenceReference, IncidentContext
from src.contracts.enums import AgentId, Environment, ResultStatus
from src.contracts.events import EventEnvelope
from src.contracts.tasks import AgentResult, AgentTask
from src.platform.orchestrator.repository import (
    SQLiteWorkflowRepository,
    WorkflowConflictError,
)
from src.platform.orchestrator.service import Orchestrator, WorkflowError
from src.platform.orchestrator.state import TaskStatus, WorkflowStage, WorkflowState

ROOT = Path(__file__).resolve().parents[2]
ANOMALY = ROOT / "contracts/examples/v1/monitoring.anomaly.detected.json"
DIAGNOSIS = ROOT / "contracts/examples/v1/diagnosis.completed.json"


def event(path: Path, **changes: object) -> EventEnvelope:
    document = json.loads(path.read_text(encoding="utf-8"))
    document.update(
        message_id=str(uuid4()),
        idempotency_key=str(uuid4()),
        timestamp=datetime.now(UTC).isoformat(),
    )
    document.update(changes)
    return EventEnvelope.model_validate(document)


class FakeContextReader:
    def __init__(self, *, environment: Environment = Environment.TEST) -> None:
        self.context = IncidentContext(
            incident_id="INC-example-001",
            version=1,
            environment=environment,
            status="OPEN",
            evidence_refs=(
                EvidenceReference(
                    evidence_id="EVD-synthetic-001",
                    incident_id="INC-example-001",
                    source="fixture",
                    object_uri="memory://fixture",
                    checksum_sha256="a" * 64,
                    observed_at=datetime.now(UTC),
                ),
            ),
        )

    async def get_incident_context(self, incident_id: str) -> IncidentContext:
        return self.context


class FakeAgent:
    def __init__(self, agent_id: AgentId, behavior: str = "success") -> None:
        self.agent_id = agent_id
        self.behavior = behavior
        self.calls = 0

    async def handle(self, task: AgentTask) -> AgentResult:
        self.calls += 1
        if self.behavior == "slow":
            await asyncio.sleep(0.05)
        if self.behavior == "hang":
            await asyncio.sleep(60)
        if self.behavior == "raise":
            raise RuntimeError("synthetic failure")
        return AgentResult(
            result_id=str(uuid4()),
            task_id="wrong" if self.behavior == "wrong" else task.task_id,
            incident_id=task.incident_id,
            agent_id=self.agent_id,
            status=(
                ResultStatus.NOT_IMPLEMENTED
                if self.behavior == "not_implemented"
                else ResultStatus.SUCCEEDED
            ),
            rationale_summary="Synthetic analysis result",
            evidence_refs=("EVD-unregistered",) if self.behavior == "unknown_evidence" else (),
        )


def setup(
    tmp_path: Path,
    agents: dict[AgentId, FakeAgent] | None = None,
    reader: FakeContextReader | None = None,
) -> tuple[Orchestrator, SQLiteWorkflowRepository]:
    repository = SQLiteWorkflowRepository(tmp_path / "checkpoints.sqlite3")
    supervisor = Orchestrator(
        repository,
        reader or FakeContextReader(),
        agents or {},
    )
    return supervisor, repository


def test_plan_dispatch_restart_and_duplicate_are_safe(tmp_path: Path) -> None:
    diagnosis = FakeAgent(AgentId.DIAGNOSIS)
    supervisor, repository = setup(tmp_path, {AgentId.DIAGNOSIS: diagnosis})
    incoming = event(ANOMALY)
    planned = asyncio.run(supervisor.ingest(incoming))
    assert planned.stage == WorkflowStage.DIAGNOSING
    assert planned.tasks[0].status == TaskStatus.PENDING
    assert "memory_pct" not in planned.tasks[0].task.task_context.signals
    assert planned.tasks[0].task.task_context.constraints == ("analysis_only", "no_side_effects")

    completed = asyncio.run(supervisor.dispatch_pending(incoming.correlation_id))
    assert completed.stage == WorkflowStage.WAITING_FOR_EVENT
    assert completed.tasks[0].status == TaskStatus.SUCCEEDED
    assert diagnosis.calls == 1

    restarted = Orchestrator(
        SQLiteWorkflowRepository(repository.path),
        FakeContextReader(),
        {AgentId.DIAGNOSIS: diagnosis},
    )
    assert asyncio.run(restarted.ingest(incoming)).revision == completed.revision
    replayed = asyncio.run(restarted.dispatch_pending(incoming.correlation_id))
    assert replayed.revision == completed.revision
    assert diagnosis.calls == 1


def test_second_event_dispatches_independent_specialists(tmp_path: Path) -> None:
    agents = {
        agent_id: FakeAgent(agent_id)
        for agent_id in (
            AgentId.DIAGNOSIS,
            AgentId.SECURITY_QUALITY,
            AgentId.RESOURCE_OPTIMIZATION,
        )
    }
    supervisor, _ = setup(tmp_path, agents)
    asyncio.run(supervisor.run_event(event(ANOMALY)))
    second = asyncio.run(supervisor.run_event(event(DIAGNOSIS)))
    assert second.stage == WorkflowStage.WAITING_FOR_EVENT
    assert len(second.receipts) == 2
    assert {item.task.agent_id for item in second.tasks} == {
        AgentId.DIAGNOSIS,
        AgentId.SECURITY_QUALITY,
    }
    assert agents[AgentId.DIAGNOSIS].calls == 1
    assert agents[AgentId.SECURITY_QUALITY].calls == 1
    assert agents[AgentId.RESOURCE_OPTIMIZATION].calls == 0
    assert second.decisions[-1].candidates[0].action_type.value == "ROLLBACK"


def test_idempotency_key_with_changed_payload_is_rejected(tmp_path: Path) -> None:
    supervisor, _ = setup(tmp_path)
    incoming = event(ANOMALY)
    asyncio.run(supervisor.ingest(incoming))
    changed = incoming.model_dump(mode="json")
    changed["message_id"] = str(uuid4())
    changed["payload"]["anomaly_score"] = 0.1
    with pytest.raises(WorkflowError, match="reused"):
        asyncio.run(supervisor.ingest(EventEnvelope.model_validate(changed)))


def test_stale_and_misdirected_events_are_rejected(tmp_path: Path) -> None:
    supervisor, repository = setup(tmp_path)
    old = event(ANOMALY, timestamp=(datetime.now(UTC) - timedelta(hours=1)).isoformat())
    with pytest.raises(WorkflowError, match="too old"):
        asyncio.run(supervisor.ingest(old))
    wrong = event(ANOMALY, target="deployment-recovery-agent")
    with pytest.raises(WorkflowError, match="not addressed"):
        asyncio.run(supervisor.ingest(wrong))
    assert repository.get(old.correlation_id) is None


def test_event_cannot_use_unknown_evidence_or_wrong_context_scope(tmp_path: Path) -> None:
    reader = FakeContextReader()
    supervisor, _ = setup(tmp_path, reader=reader)
    unknown = event(ANOMALY, evidence_refs=["EVD-unregistered"])
    with pytest.raises(WorkflowError, match="unregistered"):
        asyncio.run(supervisor.ingest(unknown))
    reader.context = reader.context.model_copy(update={"environment": Environment.PRODUCTION})
    with pytest.raises(WorkflowError, match="different scope"):
        asyncio.run(supervisor.ingest(event(ANOMALY)))


def test_diagnosis_action_must_match_incident_scope(tmp_path: Path) -> None:
    supervisor, _ = setup(tmp_path, {AgentId.DIAGNOSIS: FakeAgent(AgentId.DIAGNOSIS)})
    asyncio.run(supervisor.run_event(event(ANOMALY)))
    document = event(DIAGNOSIS).model_dump(mode="json")
    document["payload"]["recommended_actions"][0]["target"]["service"] = "another-service"
    with pytest.raises(WorkflowError, match="Candidate action"):
        asyncio.run(supervisor.ingest(EventEnvelope.model_validate(document)))


@pytest.mark.parametrize(
    ("behavior", "expected"),
    [
        ("wrong", TaskStatus.FAILED),
        ("raise", TaskStatus.FAILED),
        ("not_implemented", TaskStatus.NOT_IMPLEMENTED),
        ("unknown_evidence", TaskStatus.FAILED),
    ],
)
def test_bad_agent_outcomes_block_workflow(
    tmp_path: Path, behavior: str, expected: TaskStatus
) -> None:
    supervisor, _ = setup(tmp_path, {AgentId.DIAGNOSIS: FakeAgent(AgentId.DIAGNOSIS, behavior)})
    state = asyncio.run(supervisor.run_event(event(ANOMALY)))
    assert state.stage == WorkflowStage.BLOCKED
    assert state.tasks[0].status == expected
    with pytest.raises(WorkflowError, match="not completed"):
        asyncio.run(supervisor.ingest(event(DIAGNOSIS)))


def test_missing_agent_and_deadline_do_not_dispatch(tmp_path: Path) -> None:
    supervisor, _ = setup(tmp_path)
    state = asyncio.run(supervisor.run_event(event(ANOMALY)))
    assert state.stage == WorkflowStage.BLOCKED
    assert state.tasks[0].error_code == "AGENT_NOT_REGISTERED"

    late_dir = tmp_path / "late"
    late_dir.mkdir()
    agent = FakeAgent(AgentId.DIAGNOSIS)
    late_supervisor, _ = setup(late_dir, {AgentId.DIAGNOSIS: agent})
    planned = asyncio.run(late_supervisor.ingest(event(ANOMALY)))
    expired = planned.tasks[0].task.deadline + timedelta(seconds=1)
    late_supervisor._clock = lambda: expired
    finished = asyncio.run(late_supervisor.dispatch_pending(planned.incident_id))
    assert finished.tasks[0].status == TaskStatus.TIMED_OUT
    assert agent.calls == 0


def test_parallel_dispatch_calls_claim_each_task_once(tmp_path: Path) -> None:
    agent = FakeAgent(AgentId.DIAGNOSIS, "slow")
    supervisor, _ = setup(tmp_path, {AgentId.DIAGNOSIS: agent})
    incoming = event(ANOMALY)
    asyncio.run(supervisor.ingest(incoming))

    async def race() -> None:
        await asyncio.gather(
            supervisor.dispatch_pending(incoming.correlation_id),
            supervisor.dispatch_pending(incoming.correlation_id),
        )

    asyncio.run(race())
    assert agent.calls == 1
    assert supervisor.get(incoming.correlation_id).stage == WorkflowStage.WAITING_FOR_EVENT


def test_cancelled_dispatch_is_not_replayed_after_restart(tmp_path: Path) -> None:
    agent = FakeAgent(AgentId.DIAGNOSIS, "hang")
    supervisor, repository = setup(tmp_path, {AgentId.DIAGNOSIS: agent})
    incoming = event(ANOMALY)
    asyncio.run(supervisor.ingest(incoming))

    async def cancel_dispatch() -> None:
        worker = asyncio.create_task(supervisor.dispatch_pending(incoming.correlation_id))
        while agent.calls == 0:
            await asyncio.sleep(0)
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker

    asyncio.run(cancel_dispatch())
    restarted = Orchestrator(
        SQLiteWorkflowRepository(repository.path), FakeContextReader(), {AgentId.DIAGNOSIS: agent}
    )
    state = asyncio.run(restarted.dispatch_pending(incoming.correlation_id))
    assert state.tasks[0].status == TaskStatus.DISPATCHING
    assert agent.calls == 1


def test_sqlite_rejects_stale_checkpoint_revision(tmp_path: Path) -> None:
    supervisor, repository = setup(tmp_path)
    initial = asyncio.run(supervisor.ingest(event(ANOMALY)))
    document = initial.model_dump(mode="json")
    document["revision"] = 2
    updated = WorkflowState.model_validate(document)
    repository.replace(updated, expected_revision=1)
    with pytest.raises(WorkflowConflictError):
        repository.replace(updated, expected_revision=1)


def test_model_anomaly_routes_to_mlops_without_leaking_raw_signals(tmp_path: Path) -> None:
    agents = {
        AgentId.DIAGNOSIS: FakeAgent(AgentId.DIAGNOSIS),
        AgentId.MLOPS_LIFECYCLE: FakeAgent(AgentId.MLOPS_LIFECYCLE),
    }
    supervisor, _ = setup(tmp_path, agents)
    document = event(ANOMALY).model_dump(mode="json")
    document["payload"]["signals"] = {
        "model_id": "quiz-generator",
        "model_version": "v3",
        "private_prompt": "must not be passed to specialists",
    }
    state = asyncio.run(supervisor.run_event(EventEnvelope.model_validate(document)))
    assert {item.task.agent_id for item in state.tasks} == set(agents)
    for record in state.tasks:
        assert record.task.task_context.signals["model_id"] == "quiz-generator"
        assert "private_prompt" not in record.task.task_context.signals
    assert state.stage == WorkflowStage.WAITING_FOR_EVENT


def test_incomplete_model_identity_is_rejected(tmp_path: Path) -> None:
    supervisor, repository = setup(tmp_path)
    document = event(ANOMALY).model_dump(mode="json")
    document["payload"]["signals"] = {"model_id": "quiz-generator"}
    with pytest.raises(WorkflowError, match="incomplete model identity"):
        asyncio.run(supervisor.ingest(EventEnvelope.model_validate(document)))
    assert repository.get(document["correlation_id"]) is None


def test_duplicate_result_is_idempotent(tmp_path: Path) -> None:
    supervisor, _ = setup(tmp_path, {AgentId.DIAGNOSIS: FakeAgent(AgentId.DIAGNOSIS)})
    completed = asyncio.run(supervisor.run_event(event(ANOMALY)))
    result = completed.tasks[0].result
    assert result is not None
    repeated = asyncio.run(supervisor.record_result(result))
    assert repeated.revision == completed.revision


def test_cross_incident_evidence_in_context_is_rejected(tmp_path: Path) -> None:
    reader = FakeContextReader()
    evidence = reader.context.evidence_refs[0].model_copy(update={"incident_id": "INC-other"})
    reader.context = reader.context.model_copy(update={"evidence_refs": (evidence,)})
    supervisor, repository = setup(tmp_path, reader=reader)
    with pytest.raises(WorkflowError, match="cross-incident"):
        asyncio.run(supervisor.ingest(event(ANOMALY)))
    assert repository.get("INC-example-001") is None


def test_action_request_event_never_dispatches_execution_agent(tmp_path: Path) -> None:
    supervisor, repository = setup(tmp_path)
    fixture = ROOT / "contracts/examples/v1/deployment.action.requested.json"
    incoming = event(fixture)
    with pytest.raises(WorkflowError, match="no supported analysis route"):
        asyncio.run(supervisor.ingest(incoming))
    assert repository.get(incoming.correlation_id) is None


def test_mutated_nested_event_is_revalidated_at_boundary(tmp_path: Path) -> None:
    supervisor, repository = setup(tmp_path)
    incoming = event(ANOMALY)
    incoming.payload["anomaly_score"] = 2.0
    with pytest.raises(ValidationError):
        asyncio.run(supervisor.ingest(incoming))
    assert repository.get(incoming.correlation_id) is None


def test_agent_runtime_timeout_blocks_workflow(tmp_path: Path) -> None:
    agent = FakeAgent(AgentId.DIAGNOSIS, "hang")
    repository = SQLiteWorkflowRepository(tmp_path / "checkpoints.sqlite3")
    fixed_now = datetime.now(UTC)
    supervisor = Orchestrator(
        repository,
        FakeContextReader(),
        {AgentId.DIAGNOSIS: agent},
        clock=lambda: fixed_now,
        task_timeout=timedelta(milliseconds=20),
    )
    state = asyncio.run(supervisor.run_event(event(ANOMALY)))
    assert state.stage == WorkflowStage.BLOCKED
    assert state.tasks[0].status == TaskStatus.TIMED_OUT
    assert state.tasks[0].error_code == "AGENT_TIMEOUT"
    assert agent.calls == 1
