import asyncio

import httpx
import pytest

from src.api.app import create_app
from src.config import Settings
from src.contracts.enums import AgentId
from src.contracts.events import EventEnvelope
from src.platform.orchestrator.router import route

pytestmark = pytest.mark.integration


def request(method: str, path: str, body: dict | None = None) -> httpx.Response:
    async def send() -> httpx.Response:
        app = create_app(Settings(environment="test", _env_file=None))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as api:
            return await api.request(method, path, json=body)

    return asyncio.run(send())


def test_health_exposes_disconnected_skeleton_and_six_agents() -> None:
    assert request("GET", "/health/live").json()["status"] == "alive"
    ready = request("GET", "/health/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "skeleton-ready"
    assert ready.json()["integrations_connected"] is False
    inventory = request("GET", "/api/v1/agents").json()
    assert {item["agent_id"] for item in inventory} == {agent.value for agent in AgentId}
    assert all(item["implementation_status"] == "NOT_IMPLEMENTED" for item in inventory)


@pytest.mark.parametrize("body", [{"decision": "APPROVE"}, {"decision": "REJECT"}, {}])
def test_approval_api_never_accepts_or_executes_decisions(body: dict) -> None:
    response = request("POST", "/api/v1/approvals/APR-test/resolve", body)
    assert response.status_code == 501
    assert response.json()["code"] == "NOT_IMPLEMENTED"


def test_anomaly_routes_only_to_diagnosis(anomaly: EventEnvelope) -> None:
    assert route(anomaly) == (AgentId.DIAGNOSIS,)


def test_openapi_contains_only_scaffold_surface() -> None:
    spec = request("GET", "/openapi.json").json()
    assert "/api/v1/agents" in spec["paths"]
    assert not any("deploy" in path for path in spec["paths"])
