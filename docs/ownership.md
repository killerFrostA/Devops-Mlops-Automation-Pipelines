# Ownership and review model

| Area | Primary owner | Shared obligations |
| --- | --- | --- |
| Monitoring | Member 1 | Telemetry schema, recovery checks, evidence writes |
| Diagnosis | Member 2 | Investigation graph, RAG references, calibrated hypotheses |
| Security/quality | Member 3 | Release schema, scan evidence, blocking conditions |
| Deployment/recovery | Member 4 | Action contract, exact token verification, idempotency |
| Resource optimization | Member 5 | Bounded actions, SLO constraints, forecasts |
| MLOps | Member 6 | Model evidence, retraining/evaluation/promotion lifecycle |
| Shared platform | All six; appoint an integration lead | Contracts, orchestrator, Context, HITL, security and integration |

Actual people and GitHub handles are deliberately unassigned. Configure CODEOWNERS and branch
protection after the repository is hosted. Assign a named integration lead, a security reviewer
and an operations owner rather than assuming shared ownership alone guarantees delivery.

Hold a weekly interface review. Track each change against an acceptance criterion in the backlog.
Avoid changing contracts implicitly inside an agent implementation. A new action capability needs
an ADR, policy coverage, tool permission review and failure-path tests before execution is enabled.
