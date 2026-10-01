"""Atomic checkpoint storage; SQLite is the local durable adapter."""

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Protocol

from src.platform.orchestrator.state import WorkflowState


class WorkflowConflictError(RuntimeError):
    """A concurrent writer changed the incident checkpoint."""


class WorkflowCorruptionError(RuntimeError):
    """Stored checkpoint metadata disagrees with the validated document."""


class WorkflowRepository(Protocol):
    def get(self, incident_id: str) -> WorkflowState | None: ...

    def create(self, state: WorkflowState) -> None: ...

    def replace(self, state: WorkflowState, *, expected_revision: int) -> None: ...


class SQLiteWorkflowRepository:
    """One document per incident with compare-and-swap revisions.

    Callers must create the parent directory. Each operation uses a short-lived
    connection; a crashed dispatch remains DISPATCHING for explicit reconciliation.
    This adapter is intended for local/test use, not distributed production storage.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if str(self.path) == ":memory:":
            raise ValueError("Use a file path so checkpoints survive new connections")
        if not self.path.parent.is_dir():
            raise ValueError("Checkpoint parent directory does not exist")
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS workflow_checkpoints (
                    incident_id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL CHECK (revision >= 1),
                    state_json TEXT NOT NULL
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def get(self, incident_id: str) -> WorkflowState | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT revision, state_json FROM workflow_checkpoints WHERE incident_id = ?",
                (incident_id,),
            ).fetchone()
        if row is None:
            return None
        state = WorkflowState.model_validate_json(row[1])
        if state.incident_id != incident_id or state.revision != row[0]:
            raise WorkflowCorruptionError("Checkpoint identity or revision mismatch")
        return state

    def create(self, state: WorkflowState) -> None:
        if state.revision != 1:
            raise ValueError("New checkpoint must start at revision 1")
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    "INSERT INTO workflow_checkpoints VALUES (?, ?, ?)",
                    (state.incident_id, state.revision, state.model_dump_json()),
                )
        except sqlite3.IntegrityError as exc:
            raise WorkflowConflictError("Incident checkpoint already exists") from exc

    def replace(self, state: WorkflowState, *, expected_revision: int) -> None:
        if state.revision != expected_revision + 1:
            raise ValueError("Checkpoint revision must advance by exactly one")
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """UPDATE workflow_checkpoints
                   SET revision = ?, state_json = ?
                   WHERE incident_id = ? AND revision = ?""",
                (
                    state.revision,
                    state.model_dump_json(),
                    state.incident_id,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise WorkflowConflictError("Checkpoint revision changed")
