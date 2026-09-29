# ADR 0002: Orchestrator reads, agents append

Status: accepted for scaffolding

Operational PostgreSQL facts belong to the Context Service. Only the orchestrator reads the full
snapshot and sends bounded task context. Agents append results/evidence and query their own domain
tools. Qdrant stores curated knowledge, not current incident truth.

Separate `ContextReader` and `ContextAppender` ports make the boundary visible. Production gRPC
authorization must enforce the same boundary against authenticated workload identities.
