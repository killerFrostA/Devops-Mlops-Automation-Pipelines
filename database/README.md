# Authoritative context storage

`schema.sql` is a proposed initial PostgreSQL schema for incidents, immutable specialist results,
evidence, actions, approvals and an outbox. It is intentionally not mounted into Compose or applied.
Review it with the context/action owners and turn it into Alembic migrations when repositories exist.

Implement transaction boundaries for context version increments and result/evidence deduplication.
Use optimistic concurrency or locks for workflow writes. Persist execution reservations before
effects and handle crash-after-effect reconciliation; a unique key alone is not exactly-once execution.
Protect immutable records through application permissions and database policy, not just naming.

Redis holds hot/checkpoint/idempotency state as selected during implementation. Object storage holds
large artifacts, Qdrant curated knowledge; neither replaces authoritative incident facts.
Define retention, backups, encryption and restore verification with the actual deployment owner.
