"""Adapt Agent 1 anomaly signals to Agent 2's existing diagnosis inputs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any

from src.agents.diagnosis.contracts import DiagnosisInput


BURN_RATE_ERROR_RATE_FACTOR = 0.005

_DIRECT_ALIASES: dict[str, tuple[str, ...]] = {
    "memory_percent": ("memory_percent", "memory_pct"),
    "latency_ms": ("latency_ms", "latency_p95_ms"),
    "restart_count": ("restart_count", "pod_restarts"),
    "cpu_percent": ("cpu_percent",),
    "error_rate": ("error_rate", "error_rate_percent"),
}
_DIRECT_SIGNAL_KEYS = {alias for aliases in _DIRECT_ALIASES.values() for alias in aliases}
_KNOWN_CATEGORIES = {
    "memory_oom",
    "memory_pressure",
    "cpu_pressure",
    "error_rate",
    "latency",
}
_AGENT1_SIGNAL_KEYS = {
    "categories",
    "observed_on",
    "incidents",
    "first_seen",
    "suspects",
    "decision_reason",
    "alert_id",
    "required_evidence",
    "change_context",
    "detected_at",
}
_INCIDENT_KEYS = {
    "incident_id",
    "service_id",
    "category",
    "severity",
    "sources",
    "occurrences",
    "first_seen",
    "last_seen",
    "facts",
}
_KNOWN_FACT_KEYS = {
    "events.oom_kills",
    "events.restarts",
    "events.memory_peak_ratio",
    "events.window_seconds",
    "statistical.signal",
    "statistical.method",
    "statistical.current",
    "statistical.baseline_median",
    "statistical.slope_per_minute",
    "statistical.robust_z",
    "statistical.expected_high",
    "statistical.outside_share",
    "slo.burn_rate_long",
    "slo.burn_rate_short",
}
RATIO_TO_PERCENT = 100.0


@dataclass(frozen=True)
class SignalAdaptation:
    payload: DiagnosisInput
    limitations: tuple[str, ...]
    # True when conflicting readings were excluded, so the diagnosis rests on less data.
    degraded: bool = False


def _numeric(value: object, source: str, limitations: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        limitations.append(f"Signal '{source}' was not a finite number and was not used.")
        return None
    return float(value)


def _aware_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _category_set(signals: dict[str, Any], limitations: list[str]) -> set[str]:
    categories = signals.get("categories", [])
    if not isinstance(categories, list):
        limitations.append("Agent 1 'categories' was not a list; no category mapping was applied.")
        return set()
    known: set[str] = set()
    for category in categories:
        if not isinstance(category, str):
            limitations.append(f"Unknown Agent 1 category {category!r}; it was not interpreted.")
        elif category not in _KNOWN_CATEGORIES:
            limitations.append(f"Unknown Agent 1 category '{category}'; it was not interpreted.")
        else:
            known.add(category)
    return known


def adapt_agent1_signals(payload: DiagnosisInput) -> SignalAdaptation:
    """Return a copied diagnosis input with Agent 1 signals normalized."""
    source_signals = deepcopy(payload.signals)
    limitations: list[str] = []
    is_agent1 = "categories" in source_signals or "incidents" in source_signals
    categories = _category_set(source_signals, limitations) if is_agent1 else set()
    adapted_signals = deepcopy(source_signals)
    candidates: dict[str, list[tuple[str, float]]] = {}
    suspects = source_signals.get("suspects", [])
    if is_agent1:
        if not isinstance(suspects, list):
            limitations.append("Agent 1 'suspects' was not a list.")
        else:
            for index, suspect in enumerate(suspects):
                if not isinstance(suspect, dict):
                    limitations.append(f"Agent 1 suspect at index {index} was not an object.")
                    continue
                for key in suspect:
                    if key not in {"service_id", "probability"}:
                        limitations.append(f"Unknown Agent 1 suspect key 'suspects[{index}].{key}'.")

    def add_candidate(target: str, source: str, value: object) -> None:
        numeric = _numeric(value, source, limitations)
        if numeric is not None:
            candidates.setdefault(target, []).append((source, numeric))

    for target, aliases in _DIRECT_ALIASES.items():
        present = [(alias, source_signals[alias]) for alias in aliases if alias in source_signals]
        for alias, _ in present:
            adapted_signals.pop(alias, None)
        for alias, value in present:
            if alias == "memory_pct" and isinstance(value, (int, float)) and 0 < value < 1:
                limitations.append(
                    "Signal 'memory_pct' is between 0 and 1; it could be a fraction or a "
                    "percentage, so memory_percent was excluded."
                )
                continue
            add_candidate(target, alias, value)

    first_seen_values: list[tuple[str, datetime]] = []
    root_first_seen = source_signals.get("first_seen")
    if root_first_seen is not None and categories.intersection({"memory_oom", "memory_pressure"}):
        parsed_root_first_seen = _aware_datetime(root_first_seen)
        if parsed_root_first_seen is None:
            limitations.append(
                "Invalid or timezone-naive Agent 1 'first_seen' was not used."
            )
        else:
            first_seen_values.append(("first_seen", parsed_root_first_seen))

    incidents = source_signals.get("incidents", [])
    if is_agent1 and not isinstance(incidents, list):
        limitations.append("Agent 1 'incidents' was not a list; incident facts were not interpreted.")
        incidents = []

    for index, incident in enumerate(incidents if isinstance(incidents, list) else []):
        if not isinstance(incident, dict):
            limitations.append(f"Agent 1 incident at index {index} was not an object.")
            continue
        location = f"incidents[{index}]"
        for key in incident:
            if key not in _INCIDENT_KEYS:
                limitations.append(f"Unknown Agent 1 incident key '{location}.{key}'.")
        category = incident.get("category")
        if category is not None and (
            not isinstance(category, str) or category not in _KNOWN_CATEGORIES
        ):
            limitations.append(f"Unknown Agent 1 category '{category}' in {location}.")

        first_seen = incident.get("first_seen")
        parsed_first_seen = _aware_datetime(first_seen)
        if first_seen is not None:
            if parsed_first_seen is None:
                limitations.append(f"Invalid or timezone-naive Agent 1 '{location}.first_seen' was not used.")
            elif category in ("memory_oom", "memory_pressure"):
                first_seen_values.append((f"{location}.first_seen", parsed_first_seen))

        facts = incident.get("facts", {})
        if not isinstance(facts, dict):
            limitations.append(f"Agent 1 '{location}.facts' was not an object; facts were not interpreted.")
            continue
        for key in facts:
            if key not in _KNOWN_FACT_KEYS:
                limitations.append(f"Unknown Agent 1 fact key '{location}.facts.{key}'.")

        if "events.restarts" in facts:
            add_candidate("restart_count", f"{location}.facts.events.restarts", facts["events.restarts"])

        if "events.memory_peak_ratio" in facts:
            if category == "memory_oom":
                ratio = _numeric(
                    facts["events.memory_peak_ratio"],
                    f"{location}.facts.events.memory_peak_ratio",
                    limitations,
                )
                if ratio is not None:
                    if 0 <= ratio <= 1:
                        add_candidate(
                            "memory_percent",
                            f"{location}.facts.events.memory_peak_ratio * {RATIO_TO_PERCENT:g}",
                            ratio * RATIO_TO_PERCENT,
                        )
                        limitations.append(
                            f"Converted Agent 1 '{location}.facts.events.memory_peak_ratio' "
                            f"using the explicit ratio-to-percent factor {RATIO_TO_PERCENT:g}."
                        )
                    else:
                        limitations.append(
                            f"Agent 1 '{location}.facts.events.memory_peak_ratio' must be a "
                            "0-1 ratio and was not converted."
                        )
            else:
                limitations.append(
                    f"Agent 1 '{location}.facts.events.memory_peak_ratio' was retained as "
                    "context but not converted because the incident is not memory_oom."
                )

        for burn_key in ("slo.burn_rate_long", "slo.burn_rate_short"):
            if burn_key in facts:
                burn_rate = _numeric(
                    facts[burn_key],
                    f"{location}.facts.{burn_key}",
                    limitations,
                )
                if burn_rate is not None:
                    if burn_rate < 0:
                        limitations.append(
                            f"Agent 1 '{location}.facts.{burn_key}' was negative and not converted."
                        )
                    else:
                        add_candidate(
                            "error_rate",
                            f"{location}.facts.{burn_key} * {BURN_RATE_ERROR_RATE_FACTOR}",
                            burn_rate * BURN_RATE_ERROR_RATE_FACTOR,
                        )
                        limitations.append(
                            f"Converted Agent 1 '{location}.facts.{burn_key}' to error_rate "
                            f"using the explicit factor {BURN_RATE_ERROR_RATE_FACTOR:g}."
                        )

        if facts.get("statistical.signal") == "latency_p95_ms" and "statistical.current" in facts:
            add_candidate(
                "latency_ms",
                f"{location}.facts.statistical.current (latency_p95_ms)",
                facts["statistical.current"],
            )
        elif facts.get("statistical.signal") == "memory_mb":
            limitations.append(
                f"Agent 1 reports memory in Mo at '{location}.facts.statistical.current'; "
                "no memory limit is available, so no percentage was inferred."
            )
        elif facts.get("statistical.signal") in ("cpu_cores", "cpu_core"):
            limitations.append(
                f"Agent 1 reports CPU in cores at '{location}.facts.statistical.current'; "
                "no CPU limit is available, so no percentage was inferred."
            )
        elif facts.get("statistical.signal") is not None:
            limitations.append(
                f"Unknown Agent 1 statistical signal "
                f"'{facts.get('statistical.signal')}' in {location}; it was retained as context."
            )

        if "memory_pressure" in categories and "statistical.slope_per_minute" in facts:
            adapted_signals.setdefault(
                "statistical.slope_per_minute",
                facts["statistical.slope_per_minute"],
            )
            adapted_signals.setdefault("statistical.slope_unit", "Mo/min")

    memory_growth_observed = payload.memory_growth_observed or "memory_pressure" in categories
    updated_first_seen = payload.memory_rise_started_at
    if first_seen_values:
        limitations.append(
            "Agent 1 'first_seen' is the detection time, not the exact beginning of memory "
            "growth; it may differ by approximately 5 minutes."
        )
        distinct_times = {value for _, value in first_seen_values}
        if len(distinct_times) > 1:
            limitations.append(
                "Conflicting Agent 1 first_seen values for the memory signal; "
                "memory_rise_started_at was excluded."
            )
            updated_first_seen = None
        elif updated_first_seen is not None and updated_first_seen != first_seen_values[0][1]:
            updated_first_seen = None
            limitations.append(
                "Conflicting sources for memory_rise_started_at; the signal was excluded."
            )
        else:
            updated_first_seen = first_seen_values[0][1]

    for target, readings in candidates.items():
        normalized_values = {
            value * 100 if target == "error_rate" and 0 <= value <= 1 else value
            for _, value in readings
        }
        if len(normalized_values) > 1:
            sources = ", ".join(f"{name}={value:g}" for name, value in readings)
            limitations.append(
                f"Conflicting sources for '{target}' ({sources}); the signal was excluded."
            )
            continue
        if target == "error_rate":
            selected = readings[-1][1]
        else:
            selected = readings[0][1]
        adapted_signals[target] = selected

    if is_agent1:
        for key in source_signals:
            if (
                key not in _AGENT1_SIGNAL_KEYS
                and key not in _DIRECT_SIGNAL_KEYS
                and key != "agent1_context"
            ):
                limitations.append(f"Unknown Agent 1 signal key '{key}'.")
        adapted_signals["agent1_context"] = source_signals

    adapted_payload = payload.model_copy(
        update={
            "signals": adapted_signals,
            "memory_growth_observed": memory_growth_observed,
            "memory_rise_started_at": updated_first_seen,
        }
    )
    return SignalAdaptation(
        payload=adapted_payload,
        limitations=tuple(dict.fromkeys(limitations)),
        degraded=any(item.startswith("Conflicting") for item in limitations),
    )
