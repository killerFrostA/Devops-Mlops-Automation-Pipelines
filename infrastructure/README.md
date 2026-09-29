# Infrastructure starting points

Compose is for local development. Containers provision dependencies; pending Python adapters do
not connect automatically. Single-node Kafka uses plaintext and replication factor 1, and local
stores use development credentials. Do not extend this file directly into production.

| Profile | Services | Host ports |
| --- | --- | --- |
| Default | API | 8000 |
| `agents` | Six health-only HTTP processes | Internal 8000; no host mappings |
| `backbone` | PostgreSQL, Redis, Kafka, topic-init | 5432, 6379, 9092 |
| `intelligence` | Qdrant, MinIO | 6333, 9000, 9001 |
| `observability` | Prometheus, Grafana | 9090, 3000 |

Kafka advertises `kafka:19092` inside Compose and `localhost:9092` for local Python processes.
Topic creation is idempotent and runs after broker health. Retry/DLQ topics are provisioned, but
consumer retry handling is pending. Schema Registry is a future deployment choice, not an active
container. MLflow is an optional Python integration; its server is not provisioned here.

Prometheus initially scrapes itself. API/agent metrics instrumentation and dashboards are pending.
MinIO buckets, Qdrant collections and PostgreSQL migrations are not automatically created.

Image tags are explicit initial pins, not supply-chain verification. Validate pull/build/runtime
availability, advisories and digests before your first container release. A local Docker daemon is
required; configuration validation alone cannot verify container startup or persistence permissions.

Kubernetes provides a restricted local API template using Kustomize. Terraform declares the future
provider boundary without guessing a cloud account or deploying resources. See each directory's
README before extending it.
