# Identity, approvals and operational permissions

Each workload needs a distinct identity. Server-side Context authorization grants agents append
methods only and the orchestrator read/decision methods. Kafka ACLs grant only required topics.
Kubernetes permissions must be scoped by action class, verb, resource and namespace. The skeleton
configuration documents these policies but does not enforce them over a network yet.

## Approval implementation requirements

1. Authenticate the operator and authorize their role for the environment/action class.
2. Freeze the intent, including action ID/version, full target/parameters and preconditions.
3. Store the intent fingerprint and expiry with an authoritative approval record.
4. Resolve atomically, recording the authenticated actor, decision, reason and audit evidence.
5. Mint/verify a signed or opaque server-validated token bound to that exact intent and expiry.
6. Revalidate at execution, reserve durable idempotency, and reject changed/stale/expired intent.
7. Independently verify recovery before closing the workflow.

Reject and expiry produce no execution. A changed action needs a new approval. The dashboard
never calls Kubernetes or execution agents directly. The API placeholder cannot resolve approvals.

## Data and tools

Use managed secrets; never embed credentials in event payloads, evidence summaries or prompts.
Telemetry and ML data can contain sensitive fields, so define redaction, retention and access rules
before storing real data. No compliance regime or retention period is inferred from project cost.

Retrieved runbooks and LLM outputs are untrusted. Use typed allowlisted tools and deterministic
authorization gates. Store conclusions, scores, evidence and concise rationale, not private reasoning.

## Release controls still to configure

OIDC/service identities, mTLS, Kafka ACLs, secret manager, image digest/signature policy, dependency
and container scans, RBAC, append-only audit retention, backups/restore drills and branch protection.
CI quality checks do not constitute a security assessment.
