from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src import PROJECT_NAME, __version__
from src.agents.mlops_lifecycle.health.assessor import InsufficientSamplesError
from src.agents.mlops_lifecycle.service import MLOpsLifecycleAgent
from src.agents.registry import AGENTS
from src.config import Settings
from src.contracts.base import Contract
from src.contracts.enums import AgentId
from src.contracts.mlops import ModelHealthInput, ModelHealthReport


class HealthResponse(Contract):
    status: Literal["alive", "skeleton-ready"]
    service: str
    version: str
    runtime_mode: Literal["skeleton"] = "skeleton"
    integrations_connected: bool = False


class AgentSummary(Contract):
    agent_id: AgentId
    responsibility: str
    implementation_status: Literal["NOT_IMPLEMENTED", "PARTIAL"] = "NOT_IMPLEMENTED"


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings()
    application = FastAPI(
        title=PROJECT_NAME,
        version=__version__,
        description="Repository skeleton. Domain workflows and external integrations are pending.",
    )
    application.state.settings = config

    @application.get("/health/live", response_model=HealthResponse, tags=["health"])
    def live() -> HealthResponse:
        return HealthResponse(status="alive", service=config.service_name, version=__version__)

    @application.get("/health/ready", response_model=HealthResponse, tags=["health"])
    def ready() -> HealthResponse:
        # Readiness means the skeleton HTTP process is ready, not Kafka/DB connectivity.
        return HealthResponse(
            status="skeleton-ready", service=config.service_name, version=__version__
        )

    @application.get("/api/v1/agents", response_model=list[AgentSummary], tags=["agents"])
    def agents() -> list[AgentSummary]:
        return [
            AgentSummary(
                agent_id=agent_id,
                responsibility=agent.responsibility,
                implementation_status=agent.implementation_status,
            )
            for agent_id, agent in AGENTS.items()
        ]

    @application.post("/api/v1/approvals/{approval_id}/resolve", tags=["approvals"])
    def resolve_approval(approval_id: str, request: Request) -> JSONResponse:
        # Reserve the orchestrator-facing path without accepting unauthenticated decisions.
        return JSONResponse(
            status_code=501,
            content={
                "code": "NOT_IMPLEMENTED",
                "detail": "Approval resolution requires authenticated RBAC, durable HITL state "
                "and orchestrator integration",
            },
        )

    @application.post(
        "/api/v1/mlops/health/assess", response_model=ModelHealthReport, tags=["mlops"]
    )
    async def assess_model_health(body: ModelHealthInput) -> ModelHealthReport:
        try:
            return await MLOpsLifecycleAgent().assess_model_health(body)
        except InsufficientSamplesError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return application
