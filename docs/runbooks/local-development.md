# Local development runbook

Start the API using `python -m src.cli serve --reload`. Inspect `/health/live`, `/health/ready` and
`/api/v1/agents`. Domain implementation status should remain `NOT_IMPLEMENTED` in this scaffold.

For containers, validate `docker compose config --quiet` first. A running Docker engine is required
to build or launch. `docker compose up --build -d api` needs no database or Kafka connection.
`docker compose --profile backbone up -d` starts PostgreSQL, Redis, Kafka and a topic-init job.
Named volumes preserve local dependency data; `docker compose down` stops processes.

If the API reports ready while Kafka is unavailable, that is expected for skeleton mode.
If approvals return 501, implement the authenticated HITL/orchestrator path before changing it.
If a staging/production environment is selected, startup rejects it until the skeleton runtime is
replaced with a reviewed real implementation.

Troubleshooting:

- Python Store aliases: use an installed interpreter explicitly to create the venv.
- Import errors: install locks and editable package using the same venv interpreter.
- Schema drift: regenerate schemas, review examples and re-run contract tests.
- Docker engine unavailable: start Docker Desktop before testing container runtime.
- Port conflicts: adjust host mappings in a local Compose override.

Never replay incident events into a future production action consumer without policy and
idempotency checks. Do not use local credentials or volume layouts as production configuration.
