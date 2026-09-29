import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from src.agents.registry import AGENTS
from src.config import Settings
from src.contracts.actions import ActionIntent, ActionTarget
from src.contracts.enums import ActionType, AgentId, Environment, PolicyDecision, ResultStatus
from src.contracts.tasks import AgentTask, TaskContext
from src.platform.context.service import ContextService
from src.platform.errors import IntegrationNotConfiguredError
from src.platform.messaging.kafka import KafkaPublisher
from src.platform.orchestrator.graph import build_graph
from src.platform.policy.service import PolicyService


def task(agent_id: AgentId) -> AgentTask:
    return AgentTask(
        task_id="TASK-test",
        incident_id="INC-test",
        agent_id=agent_id,
        objective="test_boundary",
        deadline=datetime.now(UTC) + timedelta(minutes=5),
        task_context=TaskContext(service_id="fraud-api", namespace="test", environment="test"),
    )


@pytest.mark.parametrize("agent_id", [a for a in AgentId if a is not AgentId.MLOPS_LIFECYCLE])
def test_remaining_five_agents_report_unimplemented_without_confidence(agent_id: AgentId) -> None:
    result = asyncio.run(AGENTS[agent_id]().handle(task(agent_id)))
    assert result.status is ResultStatus.NOT_IMPLEMENTED
    assert result.confidence is None
    assert result.evidence_refs == ()


def test_agent_rejects_wrong_identity_and_expired_tasks() -> None:
    agent = AGENTS[AgentId.MONITORING]()
    with pytest.raises(ValueError, match="different agent"):
        asyncio.run(agent.handle(task(AgentId.DIAGNOSIS)))
    expired = task(AgentId.MONITORING).model_copy(
        update={"deadline": datetime.now(UTC) - timedelta(seconds=1)}
    )
    with pytest.raises(ValueError, match="expired"):
        asyncio.run(agent.handle(expired))


def test_fingerprint_binds_parameters_scope_and_version_and_ignores_key_order() -> None:
    intent = ActionIntent(
        action_id="ACT-test",
        incident_id="INC-test",
        action=ActionType.ROLLBACK,
        environment=Environment.TEST,
        target=ActionTarget(service="fraud-api", namespace="test"),
        parameters={"from_version": "v2", "to_version": "v1"},
        verification_profile="test-slo",
    )
    reordered = intent.model_copy(update={"parameters": {"to_version": "v1", "from_version": "v2"}})
    assert intent.fingerprint() == reordered.fingerprint()
    for changed in (
        intent.model_copy(update={"action_version": 2}),
        intent.model_copy(update={"environment": Environment.PRODUCTION}),
        intent.model_copy(update={"parameters": {"to_version": "v3"}}),
        intent.model_copy(update={"target": ActionTarget(service="other", namespace="test")}),
    ):
        assert intent.fingerprint() != changed.fingerprint()
    assert PolicyService().assess(intent).decision is PolicyDecision.BLOCK


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_skeleton_rejects_nonlocal_environment(environment: str) -> None:
    with pytest.raises(ValidationError, match="local/test"):
        Settings(environment=environment, _env_file=None)


def test_unconfigured_graph_and_context_fail_explicitly() -> None:
    with pytest.raises(IntegrationNotConfiguredError):
        build_graph()
    with pytest.raises(IntegrationNotConfiguredError):
        asyncio.run(ContextService().get_incident_context("INC-test"))


def test_kafka_does_not_silently_drop_events(anomaly) -> None:
    with pytest.raises(IntegrationNotConfiguredError):
        asyncio.run(KafkaPublisher().publish(anomaly))
