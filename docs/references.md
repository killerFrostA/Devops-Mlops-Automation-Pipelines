# Source references and design decisions

Reference material supplied by the user:

| Document | Pages | Used for |
| --- | ---: | --- |
| `BO_DSO_MultiAgent_DevOps_MLOps.pdf` | 3 | BO1–BO6, DSO1–DSO6 and evaluation ownership |
| `MultiAgent_Architecture.pdf` | 30 | Shared baseline architecture and exact-action approval binding |
| `Six_Agent_DevOps_MLOps_Technical_Architecture_v5.pdf` | 31 | Primary technical baseline and revised centralized HITL flow |

These documents describe the intended product. Statements inside them are requirements to model,
not instructions to execute infrastructure changes or implement the full system during scaffolding.
The user's scope is a complete repository skeleton. The proposed MVP in the documents remains
future implementation work.

## Conflict resolutions

- v5 takes precedence over the earlier architecture for workflow ownership.
- Human decisions go Dashboard → Orchestrator → HITL Service. The orchestrator publishes the
  `approval.resolved` event after validation; the dashboard does not publish directly to Kafka.
- `approval.requested` on Kafka is an audit/notification mirror. The future UI interaction uses the
  orchestrator-facing REST API, with WebSocket/SSE delivery to be selected during implementation.
- Context snapshots are read only by the orchestrator. Agents append results and evidence.
- Kafka remains asynchronous workflow transport; gRPC is for bounded synchronous queries.
- Exactly six agents exist. Dashboard, Context, HITL and the orchestrator are shared services.
- Canonical deployment identity is `deployment-recovery-agent`; abbreviated names used in some
  PDF examples are normalized in repository contracts.
- PDF JSON snippets are illustrative, and sometimes use abbreviated payloads. The committed
  schemas are initial normalized v1 contracts, not byte-for-byte copies of every PDF example.
- Fraud detection is the example application/model domain. The platform itself is DevOps/MLOps
  automation, so the existing downloaded application's code and model files are not imported.

## External implementation references

- [FastAPI container deployment](https://fastapi.tiangolo.com/deployment/docker/)
- [Docker Compose profiles](https://docs.docker.com/compose/how-tos/profiles/)
- [Docker Compose startup order](https://docs.docker.com/compose/how-tos/startup-order/)
- [Apache Kafka Docker quickstart](https://kafka.apache.org/41/getting-started/docker/)

Image tags and Python package pins are initial development baselines. Reassess support, advisories,
licenses and artifact digests when approving the first deployable release.
