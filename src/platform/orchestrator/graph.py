"""Optional LangGraph integration boundary; no execution nodes are implemented."""

from src.platform.errors import IntegrationNotConfiguredError


def build_graph() -> None:
    raise IntegrationNotConfiguredError(
        "Implement ingest → dispatch → aggregate → policy → HITL → execute → verify → close, "
        "with durable checkpoints, before enabling the LangGraph runtime. "
        "See docs/architecture.md and docs/roadmap.md."
    )
