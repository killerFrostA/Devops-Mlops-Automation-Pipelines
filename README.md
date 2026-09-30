# Devops-Mlops-Automation-Pipelines

**An interface-first Python repository skeleton for six specialist agents and a shared platform.**

Repository: [Devops-Mlops-Automation-Pipelines](https://github.com/killerFrostA/Devops-Mlops-Automation-Pipelines).

This repository translates the supplied architecture into ownership boundaries, typed contracts,
service entry points, development infrastructure, quality checks and a delivery plan. It is the
foundation for implementation, with a runnable local Agent 6 health-to-training demonstration.
It does not detect fraud, run production remediation, connect to Kafka, persist incident state or
authorize deployments yet.

## Start here

Python 3.12–3.14 is supported. The default is Python 3.12. Docker is optional for local infrastructure.
On Windows use PowerShell; activation is optional when invoking the virtual environment directly.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.lock
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m src.cli serve --reload
```

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/dev.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
cp .env.example .env
.venv/bin/python -m src.cli serve --reload
```

Open [API documentation](http://127.0.0.1:8000/docs),
[liveness](http://127.0.0.1:8000/health/live) or
[agent inventory](http://127.0.0.1:8000/api/v1/agents).
`/health/ready` reports readiness of the skeleton HTTP process and explicitly reports that
external integrations are disconnected. Approval resolution returns HTTP 501.

Agent 6 now supports local drift and labeled F1 assessment through
`POST /api/v1/mlops/health/assess` and a JSON CLI. The CLI can also build an assessment
from records (`--records`) or separate local baseline/prediction/label files (`--sources`).
The latter includes a synthetic machine-failure example. A separate Agent 6 demo trains,
compares and registers a candidate locally, then requests promotion review. See the Agent 6 guide.
It also has a project-neutral evaluation path for registered binary, regression and
text-generation models, with optional OpenAI/Groq rubric judging and a human-review gate.
Agent 6 reads optional LLM credentials from its own `src/agents/mlops_lifecycle/.env`;
`./.env` is reserved for platform settings.
See [Agent 6 implementation and example](src/agents/mlops_lifecycle/README.md).

## Six agents, shared coordination

| Owner | Package | Responsibility |
| --- | --- | --- |
| Member 1 | `agents/monitoring` | Telemetry, anomaly detection, recovery verification |
| Member 2 | `agents/diagnosis` | Evidence-backed root causes and recommended actions |
| Member 3 | `agents/security_quality` | Test, quality and security release assessment |
| Member 4 | `agents/deployment_recovery` | Exact, authorized deployment and recovery actions |
| Member 5 | `agents/resource_optimization` | Forecasts and bounded resource recommendations |
| Member 6 | `agents/mlops_lifecycle` | Drift, retraining, evaluation and model lifecycle |
| Shared | `platform` | Orchestrator, Context, HITL, policy, messaging, observability |

The orchestrator is a shared component, not a seventh agent. Agents receive bounded task context
and append results/evidence; only the orchestrator reads the complete operational context.

## Repository map

```text
Devops-Mlops-Automation-Pipelines/
├── src/
│   ├── agents/                 # Six replaceable specialist implementation slots
│   ├── api/                    # FastAPI app factory and local health/inventory endpoints
│   ├── contracts/              # Strict Pydantic events, tasks, actions and domain payloads
│   ├── generated/              # Protobuf output namespace (generated modules ignored)
│   └── platform/               # Shared orchestration and integration boundaries
├── contracts/
│   ├── jsonschema/v1/          # Generated, committed JSON Schemas
│   ├── protobuf/              # Context, HITL and monitoring RPC specifications
│   └── examples/v1/            # Synthetic valid event examples for independent development
├── tests/                     # Unit, contract and local integration checks
├── configs/                   # Topic, policy, RBAC and telemetry design configuration
├── infrastructure/            # Containers, Kafka bootstrap, K8s and Terraform starting points
├── database/                  # Proposed PostgreSQL schema and migration boundary
├── dashboard/                 # Approval UI structure and API integration contract
├── ml/                        # Fraud model training/evaluation/promotion interfaces
├── data/                      # Ignored datasets with tracked directory markers
├── experiments/               # Fault scenarios and metric definitions
├── docs/                      # Architecture, ADRs, ownership, backlog and runbooks
├── scripts/                   # Cross-platform development and contract generation tools
└── .github/                   # CI, dependency updates and review templates
```

## Development checks

```powershell
.\.venv\Scripts\python.exe scripts/check.py
.\.venv\Scripts\python.exe -m src.cli agents
.\.venv\Scripts\python.exe scripts/export_contracts.py --check
```

The check script runs formatting, linting, strict type checking, tests and a wheel/source build.
Contract tests verify all example messages and reject malformed events. Integration tests exercise
the local API and analysis routing; external services are not involved.

Optional integration dependencies are separated into `orchestration`, `messaging`, `storage`,
`observability` and `ml` extras. See [development](docs/development.md) for installation and locking.

## Local containers

```powershell
docker compose config --quiet
docker compose up --build -d api
docker compose --profile backbone up -d
docker compose --profile agents up --build -d
docker compose --profile intelligence --profile observability up -d
```

Profiles provision development dependencies and six health-only agent containers. They do not
connect the unfinished adapters. Published ports bind to `127.0.0.1`. Credentials in Compose are
local-only. See [infrastructure](infrastructure/README.md) for ports, profiles and limitations.

## Project documentation

- [Architecture and implementation status](docs/architecture.md)
- [Source documents and conflict decisions](docs/references.md)
- [Business / data-science objectives](docs/objectives.md)
- [Ownership and team workflow](docs/ownership.md)
- [Delivery roadmap and acceptance criteria](docs/roadmap.md)
- [Implementation backlog](docs/backlog.md)
- [Security and approval design](docs/security.md)
- [Development guide](docs/development.md)
- [Validation evidence and limits](docs/validation.md)
- [Operations runbooks](docs/runbooks/local-development.md)
- [Contributing](CONTRIBUTING.md) · [Security reporting](SECURITY.md)

No delivery budget, compliance certification or production readiness is implied by this scaffold.
Repository licensing remains an explicit project-owner decision; see [NOTICE](NOTICE).
