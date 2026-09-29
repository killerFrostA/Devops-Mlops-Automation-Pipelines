# Agent 6 local milestone validation

Validated locally on Windows with Python 3.14.2. CI is configured for Python 3.12 and 3.14 on
Windows and Linux; those remote matrix runs have not been executed in this session.

| Check | Result |
| --- | --- |
| Ruff formatting and lint | Passed |
| Strict mypy | Passed for 65 source modules |
| Pytest | 141 passed, 1 third-party MLflow/SQLAlchemy deprecation warning (writable cache under .scratch) |
| JSON Schema drift | 28 committed schemas match Python definitions |
| Event compatibility | All 11 synthetic topic examples validate against Python and JSON Schema |
| Protobuf compilation | All 4 specifications compile with namespaced imports |
| Package build | Source distribution and wheel build successfully |
| Dependency consistency | `pip check` passed |
| Agent 6 local lifecycle | Synthetic machine demo registered a review candidate; held-out champion F1 0.714, candidate F1 0.889; MLflow SQLite/run/model version verified |
| Compose configuration | All development profiles render/validate |
| Kustomize rendering | Local overlay renders 5 Kubernetes resources |

Tests cover PSI and F1 arithmetic, fixed numeric/categorical bins, delayed-label alignment,
invalid/duplicate records, delayed labels, model-window source readers, file provenance,
insufficient/absent labels, decision boundaries, Agent 6 task handling, its local HTTP/CLI paths, malformed envelopes, topic/payload mismatch,
deadlines, agent identity, missing approval proof, changed action fingerprints, blocked
continuations, explicit unavailable adapters,
deny-all policy, disconnected readiness and the six-agent inventory.

Container builds/startup, external Kafka/RPC/database integration, Terraform provider validation,
trusted data ingestion, calibrated thresholds and production acceptance have not been tested.
The local training integration is synthetic; it does not validate a production champion or promotion. Docker's engine was unavailable locally; Terraform has no configured
provider/resources. The source code and
documentation explicitly identify these implementation boundaries.

The repository ZIP includes source, fixtures, infrastructure and documentation, while excluding
the virtual environment, private source-document extracts, caches, datasets, Git internals and
build outputs. Source PDFs and the downloaded application repository are not included.
