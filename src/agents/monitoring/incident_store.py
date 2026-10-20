"""Open incidents kept on disk (SQLite), so an agent restart does not forget them.

Same interface as InMemoryIncidentStore (grouping.IncidentStore): an incident is kept by its
signature until `ttl_seconds` after its last update. After a restart, a symptom still present
updates the same incident instead of opening a duplicate one.
"""

import json
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

from src.agents.monitoring.grouping import Category, Incident
from src.contracts.enums import Severity

_SCHEMA = """
CREATE TABLE IF NOT EXISTS open_incidents (
    signature TEXT PRIMARY KEY,
    expires_at REAL NOT NULL,
    incident TEXT NOT NULL
);
"""


def _dump(incident: Incident) -> str:
    return json.dumps(
        {
            "incident_id": incident.incident_id,
            "signature": incident.signature,
            "service_id": incident.service_id,
            "namespace": incident.namespace,
            "category": incident.category.value,
            "opened_at": incident.opened_at,
            "last_seen_at": incident.last_seen_at,
            "occurrences": incident.occurrences,
            "severity": incident.severity.value,
            "sources": sorted(incident.sources),
            "facts": incident.facts,
        }
    )


def _load(text: str) -> Incident:
    data = json.loads(text)
    return Incident(
        incident_id=data["incident_id"],
        signature=data["signature"],
        service_id=data["service_id"],
        namespace=data["namespace"],
        category=Category(data["category"]),
        opened_at=data["opened_at"],
        last_seen_at=data["last_seen_at"],
        occurrences=data["occurrences"],
        severity=Severity(data["severity"]),
        sources=frozenset(data["sources"]),
        facts=data["facts"],
    )


class SqliteIncidentStore:
    def __init__(self, path: str | Path, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.executescript(_SCHEMA)

    def get(self, signature: str) -> Incident | None:
        row = self._db.execute(
            "SELECT expires_at, incident FROM open_incidents WHERE signature = ?", (signature,)
        ).fetchone()
        if row is None:
            return None
        expires_at, text = row
        if expires_at <= self._clock():
            with self._db:
                self._db.execute("DELETE FROM open_incidents WHERE signature = ?", (signature,))
            return None
        return _load(text)

    def put(self, incident: Incident, ttl_seconds: float) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO open_incidents (signature, expires_at, incident) "
                "VALUES (?, ?, ?)",
                (incident.signature, self._clock() + ttl_seconds, _dump(incident)),
            )

    def close(self) -> None:
        self._db.close()
