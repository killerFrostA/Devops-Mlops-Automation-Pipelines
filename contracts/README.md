# Contracts

Python models in `src/contracts` are the source of truth for initial JSON contracts.
Regenerate committed JSON Schemas with `python scripts/export_contracts.py`; check drift in CI.
The envelope schema includes topic/type/domain-payload coupling. Semantic checks such as deadline
ordering and action authorization also require runtime validation; JSON Schema alone cannot grant
permission to execute anything.

All eleven workflow topics have synthetic examples. Their historical timestamps are fixture data,
not instructions to replay. No example approves an action or contains a real token.

Protobuf defines bounded synchronous Context, HITL and Monitoring calls. Kafka remains the durable
workflow transport. RPC payload bytes contain the versioned JSON contract and must be validated
before use. Wire up identity, ACLs, deadlines, status codes and actual handlers during implementation.

Versioning: additive optional fields require producer/consumer compatibility review; changing
meaning, removing fields or breaking required payload shape needs a new major version. The initial
contracts are a reviewable baseline, not a negotiated long-term compatibility guarantee.
