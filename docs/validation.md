# Scaffold and orchestrator validation

Validated locally on Windows with Python 3.14.2. CI is configured for Python 3.12 and 3.14 on
Windows and Linux; those remote matrix runs have not been executed in this session.

| Check | Result |
| --- | --- |
| Ruff formatting and lint | Passed |
| Strict mypy | Passed for 62 source files |
| Pytest | 87 passed; one local pytest-cache permission warning |
| JSON Schema drift | 17 committed schemas match Python definitions |
| Event compatibility | All 11 synthetic topic examples validate against Python and JSON Schema |
| Protobuf compilation | Previously validated in scaffold baseline; not rerun for this change |
| Package build | Source distribution and wheel build successfully |
| Dependency consistency | Previously validated in scaffold baseline; not rerun for this change |
| Compose configuration | Previously validated in scaffold baseline; not rerun for this change |
| Kustomize rendering | Previously validated in scaffold baseline; not rerun for this change |

Tests cover malformed envelopes, topic/payload mismatch, deadlines, agent identity, missing
approval proof, changed action fingerprints, blocked continuations, explicit unavailable adapters,
deny-all policy, disconnected readiness and the six-agent inventory. Orchestrator tests
also cover persistent checkpoints, event idempotency, concurrent task claims, crash replay
safety, bounded signals, cross-incident evidence, deadline expiry and fail-closed results.
Recommendation tests cover diagnosis, resource and model-health routing, quality blocks,
missing evidence, action identity conflicts, optional Groq preference limits and provider fallback. Prompt tests check bounded,
versioned Jinja context and rejection of unsafe diagnosis codes and action IDs.

Container builds/startup, external Kafka/RPC/database integration, Terraform provider validation,
real algorithms, model training and production acceptance were not tested in this change.
Terraform has no configured provider/resources. The source code and documentation explicitly
identify these implementation boundaries.

The repository ZIP includes source, fixtures, infrastructure and documentation, while excluding
the virtual environment, private source-document extracts, caches, datasets, Git internals and
build outputs. Source PDFs and the downloaded application repository are not included.

The local orchestrator demo completed an anomaly-to-Diagnosis analysis, then routed a
rollback recommendation to Security/Quality and reloaded the checkpoint from SQLite.
The Groq adapter was tested with mock HTTP responses; no live Groq API call was made.
The wheel and source distribution contain the orchestrator .env.example and exclude its
private .env. The single pytest warning is from the existing
.pytest_cache directory being unwritable in this workspace; it did not affect tests.
The orchestrator's Kafka, LangGraph, Context Service, policy/HITL and deployment
integrations remain unimplemented and were not claimed as validated.
