# Agent 6 local milestone validation

Validated locally on Windows with Python 3.14.2. CI is configured for Python 3.12 and 3.14 on
Windows and Linux; those remote matrix runs have not been executed in this session.

| Check | Result |
| --- | --- |
| Ruff formatting and lint | Passed |
| Strict mypy | Passed for 57 source modules |
| Pytest | 101 passed, no warnings (writable cache under .scratch) |
| JSON Schema drift | 20 committed schemas match Python definitions |
| Event compatibility | All 11 synthetic topic examples validate against Python and JSON Schema |
| Protobuf compilation | All 4 specifications compile with namespaced imports |
| Package build | Source distribution and wheel build successfully |
| Dependency consistency | `pip check` passed |
| Compose configuration | All development profiles render/validate |
| Kustomize rendering | Local overlay renders 5 Kubernetes resources |

Tests cover PSI and F1 arithmetic, insufficient/absent labels, decision boundaries,
Agent 6 task handling, its local HTTP/CLI paths, malformed envelopes, topic/payload mismatch,
deadlines, agent identity, missing approval proof, changed action fingerprints, blocked
continuations, explicit unavailable adapters,
deny-all policy, disconnected readiness and the six-agent inventory.

Container builds/startup, external Kafka/RPC/database integration, Terraform provider validation,
trusted data ingestion, calibrated thresholds, model training and production acceptance have not
been tested. Docker's engine was unavailable locally; Terraform has no configured
provider/resources. The source code and
documentation explicitly identify these implementation boundaries.

The repository ZIP includes source, fixtures, infrastructure and documentation, while excluding
the virtual environment, private source-document extracts, caches, datasets, Git internals and
build outputs. Source PDFs and the downloaded application repository are not included.
