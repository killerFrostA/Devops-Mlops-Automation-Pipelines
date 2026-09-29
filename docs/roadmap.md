# Delivery roadmap

No schedule or allocation of the €300,000 budget is assumed. Estimate phases once staffing,
environments, data access, integrations and acceptance owners are known.

| Phase | Deliverable | Acceptance evidence |
| --- | --- | --- |
| 0 · Foundation | This scaffold, ownership and documented contracts | Local checks pass; all six domains have interfaces |
| 1 · Communication | Kafka workers, schema registry, Context/HITL gRPC, durable context | Six agents exchange validated events; unauthorized reads fail |
| 2 · Deterministic MVP | Rules/scans, policy, deployment tooling, basic drift | Incident, release and model paths work in a sandbox |
| 3 · Intelligence | RAG, anomaly models, forecasting and calibrated scoring | Improvement over deterministic baseline on held-out scenarios |
| 4 · Controlled autonomy | Dashboard, durable HITL and exact-action tokens | Approve/reject/expiry paths are auditable and enforceable |
| 5 · Reliability | Retry/DLQ, idempotency, checkpoints, circuit breakers | Duplicate delivery and dependency failure do not duplicate effects |
| 6 · Acceptance | Repeatable experiments, security/operations review | Agreed BO/DSO targets and operational readiness evidence |

## First complete vertical slice

Inject a memory-leak anomaly → persist evidence → orchestrator supplies bounded RCA task →
diagnosis returns ranked causes → policy requires approval → dashboard submits a human decision →
HITL binds the exact rollback → Agent 4 executes once → Agent 1 independently verifies →
orchestrator records closure and audit. Exercise reject, expiry, stale context, duplicates and
unavailable dependencies before adding an advanced reasoning branch.

## Definition of done

- Behavior matches a reviewed contract and acceptance scenario.
- Error paths, timeouts and authorization are tested at real trust boundaries.
- Operational telemetry and concise evidence-backed audit records exist.
- Dependency/image pins and required security checks are reviewed.
- Runbooks, rollback steps and ownership are current.
- The delivery owner approves scope, measurable outcomes and unresolved limitations.
