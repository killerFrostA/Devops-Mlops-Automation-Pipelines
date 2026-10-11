"""Decision layer: choose what to do with the current hypotheses - wait, collect more, or alert.

Each action gets an expected-utility score (expected benefit - expected cost):
- ALERT:   p * alert_benefit - (1 - p) * false_alert_cost
- WAIT:    - p * miss_cost
- COLLECT: best of the two above + information value - collect_cost, where the information
           value grows with ambiguity (probability near 0.5, or two suspects close together).
p is the probability of the top hypothesis. When COLLECT wins, the agent queries more metrics
on the suspect services, then decides again with that new evidence (confirmed or not).

Business rules come first and can override the scores (e.g. a HIGH severity incident is always
alerted). Everything is configuration: no model is trained.

With a memory, every alert sent is stored, and the operator's past verdicts on the same service
and symptom adjust the probabilities before scoring (memory.py). Business rules are not
affected: a HIGH incident is alerted even after false alerts, for safety.
"""

import dataclasses
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.agents.monitoring.grouping import Incident, most_relevant
from src.agents.monitoring.memory import AlertMemory
from src.agents.monitoring.reasoning import Hypothesis, noisy_or
from src.contracts.enums import Severity


class Action(StrEnum):
    WAIT = "WAIT"
    COLLECT_MORE = "COLLECT_MORE"
    ALERT = "ALERT"


class DecisionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alert_benefit: float = Field(default=10, gt=0)  # value of alerting a real problem
    false_alert_cost: float = Field(default=6, ge=0)  # cost of waking people for nothing
    miss_cost: float = Field(default=4, ge=0)  # cost of one more cycle without alerting
    # Cost of extra queries (load, delay). With information_value=4, collecting wins only
    # when ambiguity is above 0.5: a confident agent alerts directly.
    collect_cost: float = Field(default=2, ge=0)
    information_value: float = Field(default=4, ge=0)  # value of removing all ambiguity
    # Evidence from the extra collection, combined with the hypothesis probability.
    confirm_weight: float = Field(default=0.6, ge=0, lt=1)
    disconfirm_factor: float = Field(default=0.5, ge=0, le=1)
    # Business rules (priority over scores).
    force_alert_severities: tuple[Severity, ...] = (Severity.HIGH, Severity.CRITICAL)
    # Do not repeat the same alert. A new kind of symptom on the service, or a probability
    # rising by at least `realert_probability_increase`, is a new situation and alerts again.
    realert_after_seconds: float = Field(default=900, gt=0)
    realert_probability_increase: float = Field(default=0.15, gt=0, le=1)
    # A signal confirms the suspect when it is beyond these limits.
    confirm_error_rate: float = Field(default=0.02, ge=0)
    confirm_latency_p95_ms: float = Field(default=500, gt=0)
    confirm_memory_slope_mb_min: float = Field(default=5, gt=0)


@dataclass(frozen=True, slots=True)
class Investigation:
    service_id: str
    outcome: str  # "confirmed", "not_confirmed" or "inconclusive"
    observed: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    action: Action
    target: str | None  # most suspect service
    probability: float  # final probability after any extra collection
    scores: dict[Action, float]
    reason: str
    investigations: tuple[Investigation, ...] = ()
    notify: bool = False  # True only for an alert that was not already raised
    categories: frozenset[str] = frozenset()  # symptom kinds behind an alert
    alert_id: int | None = None  # number in the memory, to give feedback on the alert


class SignalReader(Protocol):
    """Reads current signals of a service, e.g. the Prometheus collector snapshot."""

    async def read(self, service_id: str) -> dict[str, float | None]: ...


def ambiguity(hypotheses: list[Hypothesis]) -> float:
    """1 when the top suspect is a coin flip or tied with the second, 0 when it is clear."""
    p1 = hypotheses[0].probability
    coin_flip = 1 - abs(2 * p1 - 1)  # 1 at p=0.5, 0 at p=0 or 1
    if len(hypotheses) < 2:
        return coin_flip
    tie = 1 - min(1.0, (p1 - hypotheses[1].probability) / 0.5)  # 1 equal, 0 when 0.5 apart
    return max(coin_flip, tie)


def score_actions(p: float, ambiguity_level: float, policy: DecisionPolicy) -> dict[Action, float]:
    alert = p * policy.alert_benefit - (1 - p) * policy.false_alert_cost
    wait = -p * policy.miss_cost
    collect = max(alert, wait) + policy.information_value * ambiguity_level - policy.collect_cost
    return {Action.ALERT: alert, Action.WAIT: wait, Action.COLLECT_MORE: collect}


class DecisionMaker:
    def __init__(
        self,
        policy: DecisionPolicy,
        reader: SignalReader | None = None,
        clock: Callable[[], float] = time.time,
        memory: AlertMemory | None = None,
    ) -> None:
        self.policy = policy
        self._reader = reader
        self._clock = clock
        self._memory = memory
        self._alerted: dict[str, _LastAlert] = {}  # service -> last alert raised for it

    async def decide(
        self, hypotheses: list[Hypothesis], incidents: list[Incident]
    ) -> DecisionRecord | None:
        if not hypotheses:
            return None
        record = await self._decide(hypotheses, incidents)
        if record.notify and self._memory is not None:
            assert record.target is not None
            alert_id = self._memory.record_alert(
                record.target, record.categories, record.probability, record.reason
            )
            record = dataclasses.replace(record, alert_id=alert_id)
        return record

    async def _decide(
        self, hypotheses: list[Hypothesis], incidents: list[Incident]
    ) -> DecisionRecord:
        top = hypotheses[0]
        forced = [i for i in incidents if i.severity in self.policy.force_alert_severities]
        if forced:
            worst = most_relevant(forced)
            categories: set[str] = {worst.category.value}
            if top.service_id == worst.service_id:
                categories |= _categories(top)
            return self._alert(
                worst.service_id,
                top.probability,
                {},
                f"regle metier : incident {worst.category} de gravite {worst.severity}",
                frozenset(categories),
            )

        hypotheses, learned = self._remember(hypotheses)
        top = hypotheses[0]
        scores = score_actions(top.probability, ambiguity(hypotheses), self.policy)
        best = max(scores, key=lambda a: scores[a])
        if best is Action.COLLECT_MORE and self._reader is not None:
            record = await self._collect_then_decide(hypotheses, scores)
        elif best is Action.ALERT:
            record = self._alert(
                top.service_id, top.probability, scores, "assez sur de lui", _categories(top)
            )
        else:  # WAIT, or COLLECT_MORE without any reader configured: keep watching.
            record = DecisionRecord(
                Action.WAIT, top.service_id, top.probability, scores, "pas assez d'indices"
            )
        if learned:
            record = dataclasses.replace(record, reason=f"{record.reason} ; {learned}")
        return record

    def _remember(self, hypotheses: list[Hypothesis]) -> tuple[list[Hypothesis], str]:
        """Probabilities corrected by the operator's past verdicts, and a note explaining it."""
        if self._memory is None:
            return hypotheses, ""
        adjusted: list[Hypothesis] = []
        notes: list[str] = []
        for h in hypotheses:
            p, judged = self._memory.adjust(h.service_id, _categories(h), h.probability)
            adjusted.append(Hypothesis(h.service_id, p, h.evidence))
            if judged:
                history = ", ".join(
                    f"{r.category} {r.true_alerts} vraie(s)/{r.false_alerts} fausse(s)"
                    for r in judged
                )
                notes.append(f"{h.service_id} {history} : {h.probability:.2f} -> {p:.2f}")
        adjusted.sort(key=lambda h: (-h.probability, h.service_id))
        return adjusted, ("memoire : " + " ; ".join(notes)) if notes else ""

    async def _collect_then_decide(
        self, hypotheses: list[Hypothesis], scores: dict[Action, float]
    ) -> DecisionRecord:
        assert self._reader is not None
        suspects = hypotheses[:2]
        investigations = [await self._investigate(h.service_id) for h in suspects]
        updated = [
            Hypothesis(h.service_id, self._update(h.probability, inv), h.evidence)
            for h, inv in zip(suspects, investigations, strict=True)
        ]
        updated.sort(key=lambda h: -h.probability)
        top = updated[0]
        after = score_actions(top.probability, ambiguity(updated), self.policy)
        reason = "apres collecte : " + ", ".join(
            f"{i.service_id} {i.outcome}" for i in investigations
        )
        if after[Action.ALERT] >= after[Action.WAIT]:
            record = self._alert(top.service_id, top.probability, scores, reason, _categories(top))
        else:
            record = DecisionRecord(Action.WAIT, top.service_id, top.probability, scores, reason)
        return dataclasses.replace(record, investigations=tuple(investigations))

    async def _investigate(self, service_id: str) -> Investigation:
        assert self._reader is not None
        p = self.policy
        try:
            observed = await self._reader.read(service_id)
        except Exception:  # noqa: BLE001 - a failed extra query is just inconclusive
            return Investigation(service_id, "inconclusive", {})
        checks = [
            (observed.get("error_rate"), p.confirm_error_rate),
            (observed.get("latency_p95_ms"), p.confirm_latency_p95_ms),
            (observed.get("memory_slope_mb_min"), p.confirm_memory_slope_mb_min),
        ]
        known = [(value, limit) for value, limit in checks if value is not None]
        if not known:
            outcome = "inconclusive"
        elif any(value >= limit for value, limit in known):
            outcome = "confirmed"
        else:
            outcome = "not_confirmed"
        return Investigation(service_id, outcome, observed)

    def _update(self, probability: float, investigation: Investigation) -> float:
        if investigation.outcome == "confirmed":
            return noisy_or([probability, self.policy.confirm_weight])
        if investigation.outcome == "not_confirmed":
            return probability * self.policy.disconfirm_factor
        return probability

    def _alert(
        self,
        service_id: str,
        probability: float,
        scores: dict[Action, float],
        reason: str,
        categories: frozenset[str],
    ) -> DecisionRecord:
        now = self._clock()
        last = self._alerted.get(service_id)
        why_again = None
        if last is not None:
            new = categories - last.categories
            if now - last.at >= self.policy.realert_after_seconds:
                why_again = "rappel"
            elif new:
                why_again = "nouveau symptome : " + ", ".join(sorted(new))
            elif probability >= last.probability + self.policy.realert_probability_increase:
                why_again = f"probabilite en hausse ({last.probability:.2f} -> {probability:.2f})"
        fresh = last is None or why_again is not None
        if fresh:
            seen = categories | (last.categories if last else frozenset())
            self._alerted[service_id] = _LastAlert(now, seen, probability)
        if why_again:
            reason = f"{reason} ; {why_again}"
        elif not fresh:
            reason = f"{reason} (deja alerte, situation inchangee)"
        return DecisionRecord(
            Action.ALERT,
            service_id,
            probability,
            scores,
            reason,
            notify=fresh,
            categories=categories,
        )


@dataclass(frozen=True, slots=True)
class _LastAlert:
    at: float
    categories: frozenset[str]
    probability: float


def _categories(hypothesis: Hypothesis) -> frozenset[str]:
    """Symptom kinds behind a hypothesis, e.g. {"error_rate", "memory_pressure"}."""
    return frozenset(e.category for e in hypothesis.evidence)


class SnapshotReader:
    """Adapts any `collect(service_id)` returning a MetricSnapshot-like object."""

    def __init__(self, collect: Callable[[str], Awaitable[object]]) -> None:
        self._collect = collect

    async def read(self, service_id: str) -> dict[str, float | None]:
        snapshot = await self._collect(service_id)
        return {
            name: getattr(snapshot, name, None)
            for name in ("error_rate", "latency_p95_ms", "memory_slope_mb_min", "memory_mb")
        }
