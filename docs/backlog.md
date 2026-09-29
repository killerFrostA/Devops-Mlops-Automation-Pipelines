# Implementation backlog

Every row is pending unless it explicitly says scaffolded or locally implemented. Expand estimates and assign people in
your issue tracker after approving the interface baseline.

| ID | Owner | Work | Acceptance criterion |
| --- | --- | --- | --- |
| PLAT-001 | Shared | Review initial event/protobuf contracts (scaffolded) | All six owners accept fields and compatibility rules |
| PLAT-002 | Shared | Kafka workers + registry + topic ACLs | Invalid schemas rejected; per-agent groups and partition keys verified |
| PLAT-003 | Shared | Context PostgreSQL repository and gRPC | Transactional append and evidence deduplication; orchestrator-only reads |
| PLAT-004 | Shared | LangGraph supervisor/checkpoints | Restart resumes workflow without reissuing an executed action |
| PLAT-005 | Shared | Risk/evidence/freshness policy | AUTO/APPROVAL_REQUIRED/BLOCK match an approved policy matrix |
| PLAT-006 | Shared | Authenticated HITL/token service | Scope/version/hash/expiry binding enforced; rejected/expired never execute |
| PLAT-007 | Shared | Dashboard/API operator workflow | RBAC identity used; approval sent only through orchestrator |
| PLAT-008 | Shared | Durable idempotency + outbox | Crash and duplicate delivery preserve one observable effect |
| PLAT-009 | Shared | Retry/DLQ/circuit breakers | Bounded retry-safe failures; controlled replay with action safeguards |
| PLAT-010 | Shared | Metrics/traces/security controls | Correlated telemetry, mTLS, secrets and least-privilege identities |
| A1-001 | Member 1 | Threshold monitoring + verification | Synthetic fault detected and recovery independently measured |
| A1-002 | Member 1 | Statistical/ML anomaly detector | Held-out precision/recall/delay reported against baseline |
| A2-001 | Member 2 | Deterministic RCA + tool/RAG ports | Ranked causes cite retrievable evidence and limitations |
| A2-002 | Member 2 | Constrained LLM tool reasoning | Tool allowlists and schema validation reject unauthorized suggestions |
| A3-001 | Member 3 | Tests/quality/vulnerability/secret release gate | Blocking evidence cannot be overridden by a language-model answer |
| A4-001 | Member 4 | Kubernetes/Helm deployment adapter | Exact approved action, pre/post states and idempotency verified |
| A5-001 | Member 5 | Bounded optimization rules | Replica bounds, SLO constraints and rollback conditions enforced |
| A5-002 | Member 5 | Resource forecasts | Error and savings measured without breaching SLO |
| A6-001 | Member 6 | Drift + labeled assessment, window builder and local file readers | PSI/F1 and source-backed example verified; production adapters, persisted baseline and calibration pending |
| A6-002 | Member 6 | Reproducible training/evaluation + MLflow | Local synthetic example implemented: temporal/group split, held-out comparison, SQLite MLflow run and candidate registry; real data and champion pending |
| A6-003 | Member 6 | Controlled model promotion | Local gate requests review only; release policy, HITL, canary and rollback through orchestrator/Agent 4 pending |
| OPS-001 | Shared | Provider-specific infrastructure | Reviewed environment isolation, secrets, backups and capacity |
| QA-001 | Shared | Fault experiments and acceptance report | BO/DSO metrics compared with agreed baseline |
