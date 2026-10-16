"""Layer 3: group symptoms from every detector into one incident per problem.

Signature = service_id + namespace + symptom category, with no time component: a problem that
spans 09:58 -> 10:03 stays one incident. An incident stays open while its symptom keeps being
observed, plus a cooldown, so a symptom that flickers off and on is not reopened as new.

Different categories are never merged (memory pressure and network errors at the same time are
two incidents): deciding they share a cause is diagnosis, the Diagnosis agent's job. Facts such
as a recent deployment may be attached to an incident, never interpreted.
"""

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

from pydantic import JsonValue

from src.agents.monitoring.detectors.availability import Availability, AvailabilityFinding
from src.agents.monitoring.detectors.logs import LogFinding, LogLabel
from src.agents.monitoring.detectors.slo import SLOEvaluation, SLOStatus
from src.agents.monitoring.detectors.statistical import FindingStatus, StatisticalFinding
from src.contracts.enums import Severity

_SEVERITY_RANK = {s: rank for rank, s in enumerate(Severity)}  # LOW < ... < CRITICAL


class Category(StrEnum):
    """Symptom categories. memory_oom is reserved for an actually observed OOM kill event."""

    ERROR_RATE = "error_rate"
    LATENCY = "latency"
    MEMORY_PRESSURE = "memory_pressure"
    MEMORY_OOM = "memory_oom"
    CPU_PRESSURE = "cpu_pressure"
    NETWORK_ERRORS = "network_errors"
    AUTH_ERRORS = "auth_errors"
    READINESS = "readiness"
    LOG_ERRORS = "log_errors"


@dataclass(frozen=True, slots=True)
class Symptom:
    """One observation from one detector, in a detector-independent shape."""

    service_id: str
    namespace: str
    category: Category
    source: str  # "slo", "statistical", "logs" or "events"
    severity: Severity
    facts: dict[str, JsonValue] = field(default_factory=dict)

    @property
    def signature(self) -> str:
        return f"{self.service_id}:{self.namespace}:{self.category}"


@dataclass(frozen=True, slots=True)
class Incident:
    incident_id: str
    signature: str
    service_id: str
    namespace: str
    category: Category
    opened_at: float  # unix seconds
    last_seen_at: float
    occurrences: int
    severity: Severity  # highest severity observed so far
    sources: frozenset[str]
    facts: dict[str, JsonValue]  # latest facts, keyed "<source>.<fact>"


def most_relevant(incidents: Iterable[Incident]) -> Incident:
    """The incident an alert is about: highest severity first, then the most recent one.

    With equal severity the newest incident is the new situation; an older one may only be
    the tail of a previous fault still inside the measurement windows.
    """
    order = list(Severity)
    return max(incidents, key=lambda i: (order.index(i.severity), i.opened_at, i.incident_id))


class Action(StrEnum):
    OPENED = "OPENED"
    UPDATED = "UPDATED"


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    incident: Incident
    # Publish only what changes the picture: a new incident, a higher severity, or a new
    # independent source confirming it. Repeated identical observations stay silent.
    notify: bool


class IncidentStore(Protocol):
    """Open incidents by signature. Entries expire after `ttl_seconds` without updates."""

    def get(self, signature: str) -> Incident | None: ...

    def put(self, incident: Incident, ttl_seconds: float) -> None: ...


class InMemoryIncidentStore:
    """Single-process store for tests and local runs. A Redis store with the same interface is
    needed for open incidents to survive an agent restart."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._items: dict[str, tuple[Incident, float]] = {}

    def get(self, signature: str) -> Incident | None:
        item = self._items.get(signature)
        if item is None or item[1] <= self._clock():
            self._items.pop(signature, None)
            return None
        return item[0]

    def put(self, incident: Incident, ttl_seconds: float) -> None:
        self._items[incident.signature] = (incident, self._clock() + ttl_seconds)


def new_incident_id(now: float) -> str:
    year = datetime.fromtimestamp(now, UTC).year
    return f"INC-{year}-{uuid4().hex[:10]}"


class SymptomGrouper:
    def __init__(
        self,
        store: IncidentStore,
        cooldown_seconds: float = 900,
        clock: Callable[[], float] = time.time,
        id_factory: Callable[[float], str] = new_incident_id,
    ) -> None:
        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be > 0")
        self._store = store
        self._cooldown = cooldown_seconds
        self._clock = clock
        self._id_factory = id_factory

    def ingest(self, symptoms: list[Symptom]) -> list[Decision]:
        now = self._clock()
        decisions: dict[str, Decision] = {}
        for symptom in symptoms:
            current = self._store.get(symptom.signature)
            facts = {f"{symptom.source}.{k}": v for k, v in symptom.facts.items()}
            if current is None:
                incident = Incident(
                    incident_id=self._id_factory(now),
                    signature=symptom.signature,
                    service_id=symptom.service_id,
                    namespace=symptom.namespace,
                    category=symptom.category,
                    opened_at=now,
                    last_seen_at=now,
                    occurrences=1,
                    severity=symptom.severity,
                    sources=frozenset({symptom.source}),
                    facts=facts,
                )
                decision = Decision(Action.OPENED, incident, notify=True)
            else:
                escalated = _SEVERITY_RANK[symptom.severity] > _SEVERITY_RANK[current.severity]
                new_source = symptom.source not in current.sources
                incident = replace(
                    current,
                    last_seen_at=now,
                    occurrences=current.occurrences + 1,
                    severity=symptom.severity if escalated else current.severity,
                    sources=current.sources | {symptom.source},
                    facts={**current.facts, **facts},
                )
                changed = escalated or new_source
                earlier = decisions.get(symptom.signature)
                if earlier is None:
                    decision = Decision(Action.UPDATED, incident, notify=changed)
                else:  # same signature already handled in this batch: keep one merged decision
                    decision = Decision(earlier.action, incident, notify=earlier.notify or changed)
            self._store.put(incident, self._cooldown)
            decisions[symptom.signature] = decision
        return list(decisions.values())


# --- Adapters: detector results -> symptoms. Only abnormal results become symptoms. ---

_SLO_CATEGORY = {"availability": Category.ERROR_RATE, "latency": Category.LATENCY}

_LOG_CATEGORY = {
    LogLabel.CONNECTION_REFUSED: Category.NETWORK_ERRORS,
    LogLabel.TIMEOUT: Category.NETWORK_ERRORS,
    LogLabel.AUTHENTICATION_ERROR: Category.AUTH_ERRORS,
    # A log line saying "out of memory" is not an observed OOM kill: memory_oom needs an event.
    LogLabel.OUT_OF_MEMORY: Category.MEMORY_PRESSURE,
    LogLabel.HTTP_SERVER_ERROR: Category.ERROR_RATE,
    LogLabel.READINESS_FAILURE: Category.READINESS,
    LogLabel.UNKNOWN: Category.LOG_ERRORS,
}


def from_slo(
    service_id: str, namespace: str, slo_sli: Mapping[str, str], results: list[SLOEvaluation]
) -> list[Symptom]:
    """`slo_sli` maps SLO name -> SLI kind ("availability"/"latency"), from the SLO policy."""
    return [
        Symptom(
            service_id,
            namespace,
            _SLO_CATEGORY[slo_sli[r.slo]],
            "slo",
            r.severity or Severity.MEDIUM,
            {
                "slo": r.slo,
                "rule": r.rule,
                "burn_rate_long": r.burn_rate_long,
                "burn_rate_short": r.burn_rate_short,
            },
        )
        for r in results
        if r.status is SLOStatus.BREACH
    ]


def from_statistical(
    service_id: str, namespace: str, findings: list[StatisticalFinding]
) -> list[Symptom]:
    return [
        Symptom(
            service_id,
            namespace,
            Category(f.category),
            # Chronos is its own source: when it agrees with the statistics, the incident
            # shows both sources, which is a confirmation.
            "forecast" if f.method == "forecast" else "statistical",
            f.severity or Severity.MEDIUM,
            {
                "signal": f.signal,
                "method": f.method,
                "current": f.current,
                "baseline_median": f.baseline_median,
                "robust_z": f.robust_z,
                "slope_per_minute": f.slope_per_minute,
                "expected_high": f.expected_high,
                "outside_share": f.outside_share,
            },
        )
        for f in findings
        if f.status is FindingStatus.ANOMALY
    ]


def from_availability(
    service_id: str, namespace: str, findings: list[AvailabilityFinding]
) -> list[Symptom]:
    """A service that stopped answering: readiness, HIGH (business rule: always alerted)."""
    return [
        Symptom(
            service_id,
            namespace,
            Category.READINESS,
            "availability",
            Severity.HIGH,
            {"status": f.status.value, "no_answer_for_seconds": f.window_seconds},
        )
        for f in findings
        if f.status is Availability.DOWN
    ]


def from_logs(service_id: str, namespace: str, findings: list[LogFinding]) -> list[Symptom]:
    return [
        Symptom(
            service_id,
            namespace,
            _LOG_CATEGORY[f.label],
            "logs",
            f.severity,
            {
                "kind": f.kind.value,
                "label": f.label.value,
                "template": f.template,
                "count_recent": f.count_recent,
                "count_baseline": f.count_baseline,
                "requires_review": f.requires_review,
            },
        )
        for f in findings
    ]
