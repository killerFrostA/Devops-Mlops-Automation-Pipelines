from __future__ import annotations

from src.agents.diagnosis.contracts import DiagnosisInput, EvidenceItem, EvidenceType

_AGENT1_EVIDENCE_FACTS = (
    "events.memory_peak_ratio",
    "events.restarts",
    "statistical.current",
    "statistical.baseline_median",
    "statistical.slope_per_minute",
)


def _agent1_fact_evidence(payload: DiagnosisInput) -> list[EvidenceItem]:
    context = payload.signals.get("agent1_context")
    if not isinstance(context, dict):
        return []
    incidents = context.get("incidents")
    if not isinstance(incidents, list):
        return []

    evidence: list[EvidenceItem] = []
    for index, incident in enumerate(incidents, start=1):
        if not isinstance(incident, dict):
            continue
        facts = incident.get("facts")
        if not isinstance(facts, dict):
            continue
        category = incident.get("category", "unknown")
        for fact_name in _AGENT1_EVIDENCE_FACTS:
            if fact_name not in facts:
                continue
            value = facts[fact_name]
            evidence.append(
                EvidenceItem(
                    evidence_id=(
                        f"EVD-{payload.incident_id}-AGENT1-{index:02d}-"
                        f"{fact_name.replace('.', '-')}"
                    ),
                    evidence_type="metric",
                    source="monitoring-agent",
                    provenance="provided",
                    summary=f"Agent 1 reported {fact_name}={value!r} for {category}.",
                    value={"agent1_fact_key": fact_name, fact_name: value},
                )
            )
    return evidence


def _make_evidence(
    *,
    evidence_id: str,
    evidence_type: EvidenceType,
    summary: str,
    value: dict[str, object],
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        evidence_type=evidence_type,
        source="diagnosis_input",
        provenance="derived",
        summary=summary,
        value=value,
    )


def _metric_evidence(payload: DiagnosisInput) -> list[EvidenceItem]:
    signals = payload.signals
    memory_percent = float(signals.get("memory_percent", signals.get("memory_pct", 0)) or 0)
    cpu_percent = float(signals.get("cpu_percent", 0) or 0)
    latency_ms = float(signals.get("latency_ms", 0) or 0)
    restart_count = float(signals.get("restart_count", 0) or 0)
    error_rate = float(signals.get("error_rate", 0) or 0)
    error_rate_percent = error_rate * 100 if 0 <= error_rate <= 1 else error_rate

    items: list[EvidenceItem] = []
    signal_context = {"incident_id": payload.incident_id, "service_id": payload.service_id}

    if memory_percent >= 70:
        items.append(
            _make_evidence(
                evidence_id=f"EVD-{payload.incident_id}-METRIC-01",
                evidence_type="metric",
                summary=f"Input signal reports memory usage of {memory_percent:.1f}%.",
                value={**signal_context, "memory_percent": memory_percent},
            )
        )

    if cpu_percent >= 70:
        items.append(
            _make_evidence(
                evidence_id=f"EVD-{payload.incident_id}-METRIC-02",
                evidence_type="metric",
                summary=f"Input signal reports CPU usage of {cpu_percent:.1f}%.",
                value={**signal_context, "cpu_percent": cpu_percent},
            )
        )

    if latency_ms >= 500:
        items.append(
            _make_evidence(
                evidence_id=f"EVD-{payload.incident_id}-METRIC-03",
                evidence_type="metric",
                summary=f"Input signal reports request latency of {latency_ms:.1f} ms.",
                value={**signal_context, "latency_ms": latency_ms},
            )
        )

    if restart_count > 0:
        items.append(
            _make_evidence(
                evidence_id=f"EVD-{payload.incident_id}-METRIC-04",
                evidence_type="metric",
                summary=f"Input signal reports {restart_count:g} pod restarts.",
                value={**signal_context, "restart_count": restart_count},
            )
        )

    if error_rate_percent >= 5:
        items.append(
            _make_evidence(
                evidence_id=f"EVD-{payload.incident_id}-METRIC-05",
                evidence_type="metric",
                summary=f"Input signal reports an error rate of {error_rate_percent:.1f}%.",
                value={
                    **signal_context,
                    "error_rate": error_rate,
                    "error_rate_percent": error_rate_percent,
                },
            )
        )

    return items


def _change_evidence(payload: DiagnosisInput) -> list[EvidenceItem]:
    free_text = [
        _make_evidence(
            evidence_id=f"EVD-{payload.incident_id}-GIT-{idx:02d}",
            evidence_type="git_change",
            summary=f"Input includes this recent change: {change}.",
            value={"incident_id": payload.incident_id, "change": change},
        )
        for idx, change in enumerate(payload.recent_changes[:3], start=1)
    ]
    dated_events = [
        EvidenceItem(
            evidence_id=f"EVD-{payload.incident_id}-CHANGE-{idx:02d}",
            evidence_type="git_change",
            source="diagnosis_input",
            provenance="provided",
            observed_at=event.changed_at,
            summary=f"Input records this change at {event.changed_at.isoformat()}: "
            f"{event.description}.",
            value={
                "incident_id": payload.incident_id,
                "change_id": event.change_id,
                "description": event.description,
                "changed_at": event.changed_at.isoformat(),
            },
        )
        for idx, event in enumerate(payload.recent_change_events, start=1)
    ]
    return [*free_text, *dated_events]


def _memory_growth_evidence(payload: DiagnosisInput) -> list[EvidenceItem]:
    if not payload.memory_growth_observed or payload.memory_rise_started_at is None:
        return []
    return [
        EvidenceItem(
            evidence_id=f"EVD-{payload.incident_id}-MEMORY-GROWTH",
            evidence_type="metric",
            source="diagnosis_input",
            provenance="derived",
            observed_at=payload.memory_rise_started_at,
            summary=(
                "Input records memory growth beginning at "
                f"{payload.memory_rise_started_at.isoformat()}."
            ),
            value={"memory_growth_observed": True},
        )
    ]


def collect_evidence(payload: DiagnosisInput) -> list[EvidenceItem]:
    """Return supplied evidence plus traceable evidence derived from the input only.

    This local collector does not query Prometheus, Kubernetes, Git, or log systems.
    Derived items omit confidence estimates because none is measured here. A recorded
    memory-growth item preserves the timestamp explicitly supplied in the input.
    """

    evidence = list(payload.evidence)
    evidence.extend(_metric_evidence(payload))
    evidence.extend(_agent1_fact_evidence(payload))
    evidence.extend(_change_evidence(payload))
    evidence.extend(_memory_growth_evidence(payload))

    unique: list[EvidenceItem] = []
    seen: dict[str, EvidenceItem] = {}
    for item in evidence:
        previous = seen.get(item.evidence_id)
        if previous is not None:
            if previous != item:
                raise ValueError(
                    f"Conflicting evidence records share identifier '{item.evidence_id}'"
                )
            continue
        seen[item.evidence_id] = item
        unique.append(item)
    return unique
