# Development guide

The project name is `Devops-Mlops-Automation-Pipelines`. Application packages live directly
under `src/`, and imports use `src.agents`, `src.api`, and `src.platform`.
The installed CLI is `devops-mlops`; configuration variables use the `PIDS_` prefix.

Use the README bootstrap commands. Python 3.12 is the default; 3.14 is also checked in CI.
On this Windows workstation, if `python` points to an inaccessible Store alias, use the installed
interpreter at `C:/Users/raedr/AppData/Local/Python/pythoncore-3.14-64/python.exe` to create `.venv`.
The repository itself uses portable commands and does not depend on that machine-specific path.

## Commands

| Command inside the venv | Purpose |
| --- | --- |
| `python -m src.cli serve --reload` | Local FastAPI process |
| `python -m src.cli agents` | Six ownership areas |
| `python scripts/check.py` | Format/lint/types/tests/build |
| `python scripts/export_contracts.py` | Update JSON Schema artifacts |
| `python scripts/export_contracts.py --check` | Detect stale schema artifacts |
| `python scripts/generate_examples.py` | Refresh synthetic example envelopes |
| `python scripts/generate_protos.py` | Generate RPC code after messaging tools installation |

Do not regenerate dated examples in ordinary CI. Validate their compatibility instead.

## Dependencies and locks

`pyproject.toml` declares bounded runtime dependencies and optional adapter extras.
`requirements/runtime.lock` and `requirements/dev.lock` pin the scaffold's tested runtime and
development dependency closures. `build.lock` pins packaging tools and `rpc.lock` pins Protobuf
tools. They are version pins without integrity hashes; add a
hash-enforced, multi-platform lock process before release. Optional extras are deliberately outside
these locks because the associated implementations are not active.

To work on a specific adapter: `python -m pip install -e ".[messaging]"` (or another extra).
ML libraries can require a separate interpreter/environment depending on wheel availability.
Install only the extras needed by your component; CI runs the core scaffold independently of ML.

To refresh core pins, install reviewed dependencies in a clean environment and run
`python scripts/lock_dependencies.py` (with RPC tooling installed). Review the diff and run the full check. This captures the
installed dependency closure; it is not a resolver and does not freeze optional integrations.

## Adding an agent behavior

Keep one of the six existing identities. Replace its `handle` method with bounded domain logic,
inject only the needed ports, and return a typed domain result plus execution status. Append evidence
and results through Context before publishing completion with a durable outbox strategy. Build tests
around decisions, failures and trust boundaries, not internal method calls.

## Storage and RPC

`database/schema.sql` is a proposed initial model, not an applied migration. Integrate Alembic and
review transaction/idempotency design before implementation. Protobuf files define service surfaces;
generated code is excluded and server/client adapters remain pending.
