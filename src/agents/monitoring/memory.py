"""Memory layer: remember every alert and what the operator said about it, and learn from it.

Each alert sent is stored in SQLite (one small local file, survives restarts). The operator
marks it as a true or a false alert. For each (service, symptom category) the agent keeps a
Beta distribution of its own precision:

    precision = (prior_true + true alerts) / (prior_true + prior_false + all judged alerts)

With equal priors and no feedback the precision is 0.5 and changes nothing. It then adjusts
the probability of a hypothesis before the decision, on the odds scale (Bayes' rule):

    odds(p') = odds(p) * precision / (1 - precision)     (factor bounded by max_odds_factor)

So repeated false alerts on the same service and symptom make the agent more careful there,
and confirmed ones make it more confident. Nothing is trained: two counters per pair.
"""

import math
import sqlite3
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL NOT NULL,
    service_id TEXT NOT NULL,
    probability REAL NOT NULL,
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alert_categories (
    alert_id INTEGER NOT NULL REFERENCES alerts(id),
    category TEXT NOT NULL,
    PRIMARY KEY (alert_id, category)
);
CREATE TABLE IF NOT EXISTS feedback (
    alert_id INTEGER PRIMARY KEY REFERENCES alerts(id),
    verdict TEXT NOT NULL,
    note TEXT NOT NULL,
    at REAL NOT NULL
);
"""


class MemoryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    path: str = "experiments/runs/agent1-memory.sqlite"
    # Beta prior: as if prior_true true and prior_false false alerts had already been seen.
    # Equal values = neutral; larger values = the agent needs more feedback to change.
    prior_true: float = Field(default=2, gt=0)
    prior_false: float = Field(default=2, gt=0)
    # Feedback can at most divide or multiply the odds by this factor.
    max_odds_factor: float = Field(default=4, ge=1)


class Verdict(StrEnum):
    TRUE = "vraie"
    FALSE = "fausse"


@dataclass(frozen=True, slots=True)
class Reliability:
    service_id: str
    category: str
    true_alerts: int
    false_alerts: int
    precision: float  # mean of the Beta distribution
    odds_factor: float  # applied to the odds of a hypothesis, 1 = no change


@dataclass(frozen=True, slots=True)
class StoredAlert:
    alert_id: int
    at: float
    service_id: str
    categories: tuple[str, ...]
    probability: float
    reason: str
    verdict: Verdict | None
    note: str


class AlertMemory:
    def __init__(
        self,
        path: str | Path,
        policy: MemoryPolicy | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.policy = policy or MemoryPolicy()
        self._clock = clock
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def record_alert(
        self, service_id: str, categories: Iterable[str], probability: float, reason: str
    ) -> int:
        """Store an alert that was sent; returns its number, used to give feedback."""
        with self._db:
            cursor = self._db.execute(
                "INSERT INTO alerts (at, service_id, probability, reason) VALUES (?, ?, ?, ?)",
                (self._clock(), service_id, probability, reason),
            )
            alert_id = cursor.lastrowid
            assert alert_id is not None
            self._db.executemany(
                "INSERT INTO alert_categories (alert_id, category) VALUES (?, ?)",
                [(alert_id, c) for c in sorted(set(categories))],
            )
        return alert_id

    def add_feedback(self, alert_id: int, verdict: Verdict, note: str = "") -> None:
        """The operator's judgement; giving it again replaces the previous one."""
        if self._db.execute("SELECT 1 FROM alerts WHERE id = ?", (alert_id,)).fetchone() is None:
            raise KeyError(f"Unknown alert: {alert_id}")
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO feedback (alert_id, verdict, note, at) VALUES (?, ?, ?, ?)",
                (alert_id, verdict.value, note, self._clock()),
            )

    def reliability(self, service_id: str, category: str) -> Reliability:
        rows = self._db.execute(
            "SELECT f.verdict, COUNT(*) FROM feedback f "
            "JOIN alerts a ON a.id = f.alert_id "
            "JOIN alert_categories c ON c.alert_id = a.id "
            "WHERE a.service_id = ? AND c.category = ? GROUP BY f.verdict",
            (service_id, category),
        ).fetchall()
        counts = {Verdict(verdict): count for verdict, count in rows}
        true_alerts = counts.get(Verdict.TRUE, 0)
        false_alerts = counts.get(Verdict.FALSE, 0)
        p = self.policy
        precision = (p.prior_true + true_alerts) / (
            p.prior_true + p.prior_false + true_alerts + false_alerts
        )
        # Relative to the prior, so that no feedback means exactly no change.
        prior = p.prior_true / (p.prior_true + p.prior_false)
        factor = (precision / (1 - precision)) / (prior / (1 - prior))
        factor = min(p.max_odds_factor, max(1 / p.max_odds_factor, factor))
        return Reliability(service_id, category, true_alerts, false_alerts, precision, factor)

    def adjust(
        self, service_id: str, categories: Iterable[str], probability: float
    ) -> tuple[float, list[Reliability]]:
        """Probability corrected by past feedback on this service and these symptoms.

        Several categories: geometric mean of their factors. Returns the judged ones too.
        """
        judged = [
            r
            for r in (self.reliability(service_id, c) for c in sorted(set(categories)))
            if r.true_alerts + r.false_alerts > 0
        ]
        if not judged or probability <= 0 or probability >= 1:
            return probability, judged
        factor = math.exp(sum(math.log(r.odds_factor) for r in judged) / len(judged))
        odds = probability / (1 - probability) * factor
        return odds / (1 + odds), judged

    def alerts(self, limit: int = 20) -> list[StoredAlert]:
        """Most recent alerts first, with the operator's verdict when given."""
        rows = self._db.execute(
            "SELECT a.id, a.at, a.service_id, a.probability, a.reason, f.verdict, f.note "
            "FROM alerts a LEFT JOIN feedback f ON f.alert_id = a.id "
            "ORDER BY a.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        result = []
        for alert_id, at, service_id, probability, reason, verdict, note in rows:
            categories = tuple(
                c
                for (c,) in self._db.execute(
                    "SELECT category FROM alert_categories WHERE alert_id = ? ORDER BY category",
                    (alert_id,),
                )
            )
            result.append(
                StoredAlert(
                    alert_id,
                    at,
                    service_id,
                    categories,
                    probability,
                    reason,
                    Verdict(verdict) if verdict else None,
                    note or "",
                )
            )
        return result

    def pairs(self) -> list[Reliability]:
        """Reliability of every (service, category) that has received feedback."""
        rows = self._db.execute(
            "SELECT DISTINCT a.service_id, c.category FROM feedback f "
            "JOIN alerts a ON a.id = f.alert_id "
            "JOIN alert_categories c ON c.alert_id = a.id ORDER BY 1, 2"
        ).fetchall()
        return [self.reliability(service, category) for service, category in rows]
