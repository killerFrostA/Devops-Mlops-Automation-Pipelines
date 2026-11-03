from __future__ import annotations

import re
from datetime import timedelta
from math import isfinite

from src.agents.diagnosis.contracts import (
    DiagnosisInput,
    EvidenceItem,
    EvidenceType,
    Hypothesis,
    RecentChangeEvent,
)
from src.agents.diagnosis.signatures import (
    DEPENDENCY_SIGNATURES,
    dependency_of,
    is_config_change,
    memory_limit_lowered,
    signatures_of,
)

AGENT1_MEMORY_GROWTH_MIN_SLOPE_MO_PER_MIN = 5.0
AGENT1_MEMORY_OOM_MIN_PEAK_RATIO = 0.95
AGENT1_DEPLOYMENT_DETECTION_MARGIN = timedelta(minutes=5)
# Older deployments are too weakly correlated to explain a new memory rise.
POST_DEPLOY_CORRELATION_WINDOW = timedelta(hours=6)
POST_DEPLOY_CORRELATION_BONUS = 0.1
MEMORY_GROWTH_BONUS = 0.15
SYMPTOM_ONLY_CPU_PERCENT = 50.0
SYMPTOM_ONLY_RESOURCE_FACTOR = 0.5
DEPENDENCY_SUSPECT_MIN_PROBABILITY = 0.3
TRAFFIC_SPIKE_MIN_RATIO = 1.5

_CONFLICT_THRESHOLDS: dict[str, tuple[float, ...]] = {
    "memory_percent": (70.0,),
    "cpu_percent": (50.0, 70.0),
    "restart_count": (2.0,),
    "latency_ms": (500.0, 800.0),
    "error_rate_percent": (5.0, 10.0),
}


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

    if cpu_percent >= 50:
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


def _canonical_metric(name: str) -> str:
    if name in ("memory_percent", "memory_pct"):
        return "memory_percent"
    if name == "error_rate":
        return "error_rate_percent"
    return name


def _numeric_value(value: object, metric: str) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not isfinite(numeric):
        return None
    if metric == "error_rate_percent" and 0 <= numeric <= 1:
        return numeric * 100
    return numeric


def has_oom_observation(item: EvidenceItem) -> bool:
    text = f"{item.summary} {item.value.get('reason', '')}".lower()
    if re.search(r"\boom\b|\boomkilled\b|out[- ]of[- ]memory", text) is None:
        return False
    return (
        re.search(
            r"\b(?:no|not|without|never)\s+(?:an?\s+)?(?:evidence of\s+)?"
            r"(?:oom\b|oomkilled\b|out[- ]of[- ]memory)",
            text,
        )
        is None
    )


def conflicting_metrics(payload: DiagnosisInput) -> tuple[str, ...]:
    """Return metrics whose supplied readings fall on different rule threshold bands."""
    readings: dict[str, list[float]] = {}
    for name, value in payload.signals.items():
        metric = _canonical_metric(name)
        if metric in _CONFLICT_THRESHOLDS:
            numeric = _numeric_value(value, metric)
            if numeric is not None:
                readings.setdefault(metric, []).append(numeric)

    for item in payload.evidence:
        for name, value in item.value.items():
            metric = _canonical_metric(name)
            if metric in _CONFLICT_THRESHOLDS:
                numeric = _numeric_value(value, metric)
                if numeric is not None:
                    readings.setdefault(metric, []).append(numeric)

    conflicts = []
    for metric, values in readings.items():
        thresholds = _CONFLICT_THRESHOLDS[metric]
        bands = {tuple(value >= threshold for threshold in thresholds) for value in values}
        if len(bands) > 1:
            conflicts.append(metric)
    return tuple(sorted(conflicts))


def collect_evidence(payload: DiagnosisInput) -> list[EvidenceItem]:
    """Return supplied evidence plus traceable evidence derived from the input only.

    This local collector does not query Prometheus, Kubernetes, Git, or log systems.
    Derived items intentionally omit observation times and confidence estimates because
    neither is measured by this component.
    """

    evidence = list(payload.evidence)
    evidence.extend(_metric_evidence(payload))
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


def _measure(
    payload: DiagnosisInput,
    *names: str,
    conflicts: tuple[str, ...] = (),
) -> float:
    if any(_canonical_metric(name) in conflicts for name in names):
        return 0.0
    for name in names:
        if name in payload.signals:
            value = _numeric_value(payload.signals[name], _canonical_metric(name))
            if value is not None:
                return value
    for evidence in payload.evidence:
        for name in names:
            if name in evidence.value:
                value = _numeric_value(evidence.value[name], _canonical_metric(name))
                if value is not None:
                    return value
    return 0.0


def signal_conflicts(payload: DiagnosisInput) -> tuple[str, ...]:
    """Explain disagreements that change the interpretation of a rule threshold."""
    return tuple(
        f"Conflicting readings for {metric}; candidates relying on this metric were not "
        "ranked from a single selected value."
        for metric in conflicting_metrics(payload)
    )


def _relevant_evidence(
    evidence: tuple[EvidenceItem, ...] | list[EvidenceItem],
    *,
    cause: str,
    conflicts: tuple[str, ...] = (),
) -> tuple[str, ...]:
    references = []
    for item in evidence:
        if cause in ("MEMORY_PRESSURE", "MEMORY_LEAK_POST_DEPLOY"):
            memory = _numeric_value(
                item.value.get("memory_percent", item.value.get("memory_pct")),
                "memory_percent",
            )
            restarts = _numeric_value(item.value.get("restart_count"), "restart_count")
            has_oom = has_oom_observation(item)
            supports = memory is not None and memory >= 70
            if cause == "MEMORY_LEAK_POST_DEPLOY":
                supports = supports or (restarts is not None and restarts > 0) or has_oom
                supports = supports or item.value.get("memory_growth_observed") is True
        else:
            cpu = (
                None
                if "cpu_percent" in conflicts
                else _numeric_value(item.value.get("cpu_percent"), "cpu_percent")
            )
            latency = (
                None
                if "latency_ms" in conflicts
                else _numeric_value(item.value.get("latency_ms"), "latency_ms")
            )
            error_rate = (
                None
                if "error_rate_percent" in conflicts
                else _numeric_value(item.value.get("error_rate"), "error_rate_percent")
            )
            if error_rate is None and "error_rate_percent" not in conflicts:
                error_rate = _numeric_value(
                    item.value.get("error_rate_percent"), "error_rate_percent"
                )
            supports = (
                (cpu is not None and cpu >= 70)
                or (error_rate is not None and error_rate >= 10)
                or (
                    latency is not None
                    and latency >= 800
                    and (
                        (cpu is not None and cpu >= 50)
                        or (error_rate is not None and error_rate >= 5)
                    )
                )
            )
        if supports:
            references.append(item.evidence_id)
    return tuple(dict.fromkeys(references))


def _memory_hypothesis(
    payload: DiagnosisInput,
    cause: str,
    conflicts: tuple[str, ...] = (),
) -> Hypothesis:
    memory_percent = _measure(payload, "memory_percent", "memory_pct")
    restart_count = _measure(payload, "restart_count")
    latency_ms = _measure(payload, "latency_ms")

    score = 0.2
    score += min(memory_percent / 100.0, 1.0) * 0.35
    score += min(restart_count / 5.0, 1.0) * 0.2
    score += 0.1 if latency_ms >= 500 else 0.0
    evidence_refs = list(
        _relevant_evidence(
            payload.evidence,
            cause=cause,
            conflicts=conflicts,
        )
    )
    if cause == "MEMORY_LEAK_POST_DEPLOY":
        if payload.memory_growth_observed:
            growth_ref = f"EVD-{payload.incident_id}-MEMORY-GROWTH"
            if any(item.evidence_id == growth_ref for item in payload.evidence):
                evidence_refs.append(growth_ref)
                # A measured growth is the core evidence of a leak; caught early (before OOM
                # restarts or latency impact) the other terms are still low (lab: 0.57).
                score += MEMORY_GROWTH_BONUS
        qualifying_refs = _qualifying_deployment_refs(payload)
        evidence_refs.extend(qualifying_refs)
        if qualifying_refs:
            score += POST_DEPLOY_CORRELATION_BONUS
    if cause == "MEMORY_LIMIT_REACHED":
        limit_refs = _memory_limit_cut_refs(payload)
        evidence_refs.extend(limit_refs)
        if limit_refs:
            score += POST_DEPLOY_CORRELATION_BONUS
    return Hypothesis(
        rank=1,
        code=cause,
        probability=round(min(score, 1.0), 2),
        evidence_refs=tuple(dict.fromkeys(evidence_refs)),
    )


def _resource_pressure_hypothesis(
    payload: DiagnosisInput,
    conflicts: tuple[str, ...] = (),
) -> Hypothesis:
    memory_percent = _measure(payload, "memory_percent", "memory_pct")
    cpu_percent = _measure(payload, "cpu_percent")
    latency_ms = _measure(payload, "latency_ms")
    error_rate = _measure(payload, "error_rate")

    score = 0.3
    score += min(cpu_percent / 100.0, 1.0) * 0.4
    score += 0.15 if latency_ms >= 500 else 0.0
    score += min(error_rate / 10.0, 1.0) * 0.25
    score += min(memory_percent / 100.0, 1.0) * 0.1
    evidence_refs = _relevant_evidence(
        payload.evidence,
        cause="RESOURCE_PRESSURE_OR_CPU_STARVATION",
        conflicts=conflicts,
    )
    return Hypothesis(
        rank=1,
        code="RESOURCE_PRESSURE_OR_CPU_STARVATION",
        probability=round(min(score, 1.0), 2),
        evidence_refs=evidence_refs,
    )


def _insufficient_evidence_hypothesis() -> Hypothesis:
    return Hypothesis(
        rank=1,
        code="INSUFFICIENT_EVIDENCE",
        probability=0.25,
        evidence_refs=(),
    )


def _agent1_fact_measure(payload: DiagnosisInput, fact_name: str) -> float | None:
    for item in payload.evidence:
        value = item.value.get(fact_name)
        numeric = _numeric_value(value, fact_name)
        if numeric is not None:
            return numeric
    return None


def _agent1_fact_refs(payload: DiagnosisInput, *fact_names: str) -> tuple[str, ...]:
    requested = set(fact_names)
    return tuple(
        item.evidence_id
        for item in payload.evidence
        if item.value.get("agent1_fact_key") in requested
    )


def _agent1_hypothesis(
    payload: DiagnosisInput,
    *,
    code: str,
    fact_names: tuple[str, ...],
    probability: float,
) -> Hypothesis:
    return Hypothesis(
        rank=1,
        code=code,
        probability=probability,
        evidence_refs=_agent1_fact_refs(payload, *fact_names),
    )


def _agent1_memory_category(payload: DiagnosisInput, category: str) -> bool:
    context = payload.signals.get("agent1_context")
    if not isinstance(context, dict):
        return False
    categories = context.get("categories")
    return isinstance(categories, list) and category in categories


def _is_deployment_event(event: RecentChangeEvent) -> bool:
    return bool(
        re.search(
            r"\bdeploy(?:ment|ed)?\b",
            f"{event.change_id} {event.description}",
            flags=re.IGNORECASE,
        )
    )


def qualifies_as_prior_deployment(payload: DiagnosisInput, event: RecentChangeEvent) -> bool:
    """A deployment between 5 minutes and 6 hours before the recorded memory rise."""
    if payload.memory_rise_started_at is None or not _is_deployment_event(event):
        return False
    lead = payload.memory_rise_started_at - event.changed_at
    return AGENT1_DEPLOYMENT_DETECTION_MARGIN <= lead <= POST_DEPLOY_CORRELATION_WINDOW


def _qualifying_deployment_precedes_memory_rise(payload: DiagnosisInput) -> bool:
    return any(
        qualifies_as_prior_deployment(payload, event) for event in payload.recent_change_events
    )


def _qualifying_deployment_refs(payload: DiagnosisInput) -> tuple[str, ...]:
    return tuple(
        f"EVD-{payload.incident_id}-CHANGE-{index:02d}"
        for index, event in enumerate(payload.recent_change_events, start=1)
        if qualifies_as_prior_deployment(payload, event)
    )


def _uncertain_deployment_correlation(payload: DiagnosisInput) -> bool:
    if payload.memory_rise_started_at is None:
        return False
    return any(
        _is_deployment_event(event)
        and timedelta(0)
        < payload.memory_rise_started_at - event.changed_at
        < AGENT1_DEPLOYMENT_DETECTION_MARGIN
        for event in payload.recent_change_events
    )


def _memory_limit_cut_refs(payload: DiagnosisInput) -> tuple[str, ...]:
    """Change records (commits, configs, rollouts) that lowered the memory limit."""
    return tuple(
        item.evidence_id
        for item in payload.evidence
        if item.evidence_type in ("git_change", "config", "k8s_event")
        and memory_limit_lowered(item)
    )


def _refs_with_signature(payload: DiagnosisInput, *names: str) -> tuple[str, ...]:
    wanted = set(names)
    return tuple(item.evidence_id for item in payload.evidence if wanted & signatures_of(item))


def _suspected_dependency(payload: DiagnosisInput) -> str | None:
    """Another service named by evidence or by Agent 1 suspects."""
    for item in payload.evidence:
        dependency = dependency_of(item)
        if dependency is not None and dependency != payload.service_id:
            return dependency
    context = payload.signals.get("agent1_context")
    suspects = context.get("suspects") if isinstance(context, dict) else None
    if isinstance(suspects, list):
        for suspect in suspects:
            if not isinstance(suspect, dict):
                continue
            service = suspect.get("service_id")
            probability = suspect.get("probability")
            if (
                isinstance(service, str)
                and service != payload.service_id
                and isinstance(probability, (int, float))
                and not isinstance(probability, bool)
                and probability >= DEPENDENCY_SUSPECT_MIN_PROBABILITY
            ):
                return service
    return None


def _crash_loop_hypothesis(payload: DiagnosisInput) -> Hypothesis | None:
    crash_refs = _refs_with_signature(payload, "crash_loop")
    if not crash_refs or _refs_with_signature(payload, "oom"):
        # An OOM-driven crash loop is handled by the memory rules.
        return None
    config_log_refs = _refs_with_signature(payload, "config_invalid")
    config_change_refs = tuple(
        item.evidence_id for item in payload.evidence if is_config_change(item)
    )
    if config_log_refs:
        code, probability = "CONFIG_ERROR_CRASHLOOP", 0.85
    elif config_change_refs:
        code, probability = "CONFIG_ERROR_CRASHLOOP", 0.7
    else:
        code, probability = "CRASH_LOOP_UNEXPLAINED", 0.55
    return Hypothesis(
        rank=1,
        code=code,
        probability=probability,
        evidence_refs=tuple(dict.fromkeys((*crash_refs, *config_log_refs, *config_change_refs))),
    )


def _dependency_hypotheses(
    payload: DiagnosisInput,
    *,
    error_rate: float,
    latency_ms: float,
) -> list[Hypothesis]:
    hypotheses = []
    pool_refs = _refs_with_signature(payload, "pool_exhausted")
    if pool_refs:
        hypotheses.append(
            Hypothesis(
                rank=1,
                code="CONNECTION_POOL_EXHAUSTION",
                probability=0.85 if latency_ms >= 500 else 0.8,
                evidence_refs=pool_refs,
            )
        )
    dependency_refs = _refs_with_signature(payload, *sorted(DEPENDENCY_SIGNATURES))
    if dependency_refs:
        hypotheses.append(
            Hypothesis(
                rank=1,
                code="DEPENDENCY_UNAVAILABLE",
                probability=0.83 if error_rate >= 5 else 0.78,
                evidence_refs=dependency_refs,
            )
        )
    elif not pool_refs and error_rate >= 5 and _suspected_dependency(payload) is not None:
        # Agent 1 suspects another service but no log confirms it yet.
        hypotheses.append(
            Hypothesis(
                rank=1,
                code="DEPENDENCY_UNAVAILABLE",
                probability=0.5,
                evidence_refs=_relevant_evidence(
                    payload.evidence, cause="RESOURCE_PRESSURE_OR_CPU_STARVATION"
                ),
            )
        )
    return hypotheses


def _traffic_spike_hypothesis(payload: DiagnosisInput, cpu_percent: float) -> Hypothesis | None:
    for item in payload.evidence:
        rps = _numeric_value(item.value.get("rps"), "rps")
        baseline = _numeric_value(item.value.get("rps_baseline"), "rps")
        if rps is None or baseline is None or baseline <= 0:
            continue
        ratio = rps / baseline
        if ratio < TRAFFIC_SPIKE_MIN_RATIO:
            continue
        score = 0.55 + 0.25 * min((ratio - 1) / 2, 1.0) + (0.1 if cpu_percent >= 70 else 0.0)
        return Hypothesis(
            rank=1,
            code="TRAFFIC_SPIKE",
            probability=round(min(score, 1.0), 2),
            evidence_refs=(item.evidence_id,),
        )
    return None


def evaluate_hypotheses(payload: DiagnosisInput) -> list[Hypothesis]:
    """Rank deterministic heuristic candidates; scores are not calibrated probabilities."""

    conflicts = conflicting_metrics(payload)
    memory_percent = _measure(payload, "memory_percent", "memory_pct", conflicts=conflicts)
    restart_count = _measure(payload, "restart_count", conflicts=conflicts)
    cpu_percent = _measure(payload, "cpu_percent", conflicts=conflicts)
    latency_ms = _measure(payload, "latency_ms", conflicts=conflicts)
    error_rate = _measure(payload, "error_rate", conflicts=conflicts)

    hypotheses: list[Hypothesis] = []
    agent1_oom_ratio = (
        _agent1_fact_measure(payload, "events.memory_peak_ratio")
        if _agent1_memory_category(payload, "memory_oom")
        else None
    )
    if agent1_oom_ratio is not None and agent1_oom_ratio >= AGENT1_MEMORY_OOM_MIN_PEAK_RATIO:
        hypotheses.append(
            _agent1_hypothesis(
                payload,
                code="MEMORY_LIMIT_REACHED",
                fact_names=("events.memory_peak_ratio", "events.restarts"),
                probability=0.8,
            )
        )

    agent1_growth_slope = (
        _agent1_fact_measure(payload, "statistical.slope_per_minute")
        if _agent1_memory_category(payload, "memory_pressure")
        else None
    )
    if (
        payload.memory_growth_observed
        and agent1_growth_slope is not None
        and agent1_growth_slope >= AGENT1_MEMORY_GROWTH_MIN_SLOPE_MO_PER_MIN
    ):
        cause = (
            "MEMORY_LEAK_POST_DEPLOY"
            if _qualifying_deployment_precedes_memory_rise(payload)
            else "MEMORY_GROWTH"
        )
        growth = _agent1_hypothesis(
            payload,
            code=cause,
            fact_names=(
                "statistical.current",
                "statistical.baseline_median",
                "statistical.slope_per_minute",
            ),
            probability=0.7,
        )
        if cause == "MEMORY_LEAK_POST_DEPLOY":
            growth = growth.model_copy(
                update={
                    "evidence_refs": tuple(
                        dict.fromkeys(
                            (*growth.evidence_refs, *_qualifying_deployment_refs(payload))
                        )
                    )
                }
            )
        hypotheses.append(growth)

    if (
        not hypotheses
        and not {"memory_percent", "restart_count"}.intersection(conflicts)
        and memory_percent >= 70
    ):
        has_memory_failure_signal = (
            payload.memory_growth_observed
            or restart_count > 0
            or any(has_oom_observation(item) for item in payload.evidence)
        )
        qualifying_change = _qualifying_deployment_precedes_memory_rise(payload)
        if (
            has_memory_failure_signal
            and _memory_limit_cut_refs(payload)
            and not payload.memory_growth_observed
        ):
            # The release lowered the limit and memory did not grow: the workload no longer
            # fits, it does not leak (raise the limit or revert the change, no restart loop).
            memory_cause = "MEMORY_LIMIT_REACHED"
        elif has_memory_failure_signal and qualifying_change:
            memory_cause = "MEMORY_LEAK_POST_DEPLOY"
        else:
            memory_cause = "MEMORY_PRESSURE"
        hypotheses.append(_memory_hypothesis(payload, memory_cause, conflicts))

    crash_loop = _crash_loop_hypothesis(payload)
    if crash_loop is not None:
        hypotheses.append(crash_loop)
    hypotheses.extend(_dependency_hypotheses(payload, error_rate=error_rate, latency_ms=latency_ms))
    traffic_spike = _traffic_spike_hypothesis(payload, cpu_percent)
    if traffic_spike is not None:
        hypotheses.append(traffic_spike)

    resource_has_multiple_signals = (
        cpu_percent >= 70
        or error_rate >= 10
        or (latency_ms >= 800 and (cpu_percent >= 50 or error_rate >= 5))
    )
    if resource_has_multiple_signals:
        candidate = _resource_pressure_hypothesis(payload, conflicts)
        if candidate.probability >= 0.55:
            post_deploy_leak = any(item.code == "MEMORY_LEAK_POST_DEPLOY" for item in hypotheses)
            if hypotheses and (cpu_percent < SYMPTOM_ONLY_CPU_PERCENT or post_deploy_leak):
                # Errors and latency are already explained by another candidate; without a
                # CPU measurement, or next to a dated post-deployment leak (whose GC and
                # contention raise CPU), CPU starvation stays a lower-ranked alternative.
                candidate = candidate.model_copy(
                    update={
                        "probability": round(
                            candidate.probability * SYMPTOM_ONLY_RESOURCE_FACTOR, 2
                        )
                    }
                )
            hypotheses.append(candidate)

    if not hypotheses:
        return [_insufficient_evidence_hypothesis()]

    ranked = sorted(hypotheses, key=lambda item: (-item.probability, item.code))
    return [
        Hypothesis(
            rank=index,
            code=hypothesis.code,
            probability=hypothesis.probability,
            evidence_refs=hypothesis.evidence_refs,
        )
        for index, hypothesis in enumerate(ranked, start=1)
    ]
