"""Reasoning layer: rank which service most probably originates the incidents seen together.

Input: incidents active in a recent window (from the grouping layer) and a dependency graph.
For each service, evidence is collected:
- own symptom: an incident on the service itself;
- dependent symptom: an incident on a service that depends on it (the fault may come from
  below and propagate upward);
- dependency log hint: a log template on a caller that points at its dependencies
  (e.g. "connection pool exhausted").
Evidence is combined with noisy-OR: P = 1 - prod(1 - w_i). Each piece of evidence is an
independent chance of explaining the problem; agreeing evidence raises the probability.

Output: hypotheses ranked by probability, each with the evidence used. They are hypotheses
handed to the Diagnosis agent, not a final root-cause verdict. No model is trained: weights
come from configuration.
"""

import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.agents.monitoring.grouping import Category, Incident


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ServiceNode(_Config):
    depends_on: tuple[str, ...] = ()


class LogHint(_Config):
    pattern: str
    weight: float = Field(gt=0, lt=1)

    @field_validator("pattern")
    @classmethod
    def valid_regex(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as error:
            raise ValueError(f"invalid regex {value!r}: {error}") from error
        return value


class ReasoningConfig(_Config):
    window_seconds: float = Field(default=300, gt=0)
    services: dict[str, ServiceNode] = {}
    own_symptom_weights: dict[Category, float] = {}
    dependent_symptom_weight: float = Field(default=0.35, ge=0, lt=1)
    dependency_log_hints: tuple[LogHint, ...] = ()
    # Explaining away: a symptom also present on a service this one calls is probably
    # inherited from it, so it weighs less against this service. Without it, in a chain
    # checkout -> payment -> db with a slow db, payment ranked as high as db (one-hop only).
    explained_away_factor: float = Field(default=0.5, gt=0, le=1)

    @model_validator(mode="after")
    def coherent(self) -> "ReasoningConfig":
        for name, node in self.services.items():
            unknown = [d for d in node.depends_on if d not in self.services]
            if unknown:
                raise ValueError(f"{name} depends on unknown service(s): {unknown}")
            if name in node.depends_on:
                raise ValueError(f"{name} cannot depend on itself")
        for category, weight in self.own_symptom_weights.items():
            if not 0 <= weight < 1:
                raise ValueError(f"weight for {category} must be in [0, 1)")
        return self

    def dependencies(self, service_id: str) -> tuple[str, ...]:
        node = self.services.get(service_id)
        return node.depends_on if node else ()


def load_reasoning_config(path: Path) -> ReasoningConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ReasoningConfig.model_validate(data)


@dataclass(frozen=True, slots=True)
class Evidence:
    kind: str  # "own_symptom", "dependent_symptom" or "dependency_log_hint"
    observed_on: str  # service where the symptom was observed
    category: str
    weight: float
    detail: str


@dataclass(frozen=True, slots=True)
class Hypothesis:
    service_id: str
    probability: float
    evidence: tuple[Evidence, ...]


def noisy_or(weights: list[float]) -> float:
    """Probability that at least one independent cause is active."""
    return 1 - math.prod(1 - w for w in weights)


class Reasoner:
    """Keeps incidents seen in the recent window and ranks suspect services."""

    def __init__(self, config: ReasoningConfig, clock: Callable[[], float] = time.time) -> None:
        self.config = config
        self._clock = clock
        self._recent: dict[str, Incident] = {}

    def observe(self, incidents: list[Incident]) -> None:
        for incident in incidents:
            self._recent[incident.incident_id] = incident

    def rank(self) -> list[Hypothesis]:
        horizon = self._clock() - self.config.window_seconds
        self._recent = {k: v for k, v in self._recent.items() if v.last_seen_at >= horizon}
        evidence: dict[str, list[Evidence]] = {}
        seen = {(i.service_id, i.category) for i in self._recent.values()}
        for incident in self._recent.values():
            for target, item in self._evidence_from(incident, seen):
                evidence.setdefault(target, []).append(item)
        hypotheses = [
            Hypothesis(service, noisy_or([e.weight for e in items]), tuple(items))
            for service, items in evidence.items()
        ]
        return sorted(hypotheses, key=lambda h: (-h.probability, h.service_id))

    def _evidence_from(
        self, incident: Incident, seen: set[tuple[str, Category]]
    ) -> list[tuple[str, Evidence]]:
        cfg = self.config
        observed = incident.service_id
        found: list[tuple[str, Evidence]] = []
        dependencies = cfg.dependencies(observed)
        own = cfg.own_symptom_weights.get(incident.category, 0.0)
        inherited = [d for d in dependencies if (d, incident.category) in seen]
        detail = "sur le service"
        if inherited:
            own *= cfg.explained_away_factor
            detail = f"sur le service, aussi vu sur {', '.join(inherited)}"
        if own > 0:
            found.append(
                (observed, Evidence("own_symptom", observed, incident.category, own, detail))
            )
        if cfg.dependent_symptom_weight > 0:
            for dependency in dependencies:
                found.append(
                    (
                        dependency,
                        Evidence(
                            "dependent_symptom",
                            observed,
                            incident.category,
                            cfg.dependent_symptom_weight,
                            f"{observed} depend de {dependency}",
                        ),
                    )
                )
        template = incident.facts.get("logs.template")
        if isinstance(template, str) and dependencies:
            for hint in cfg.dependency_log_hints:
                if re.search(hint.pattern, template, re.IGNORECASE):
                    for dependency in dependencies:
                        found.append(
                            (
                                dependency,
                                Evidence(
                                    "dependency_log_hint",
                                    observed,
                                    incident.category,
                                    hint.weight,
                                    f"log : {template}",
                                ),
                            )
                        )
                    break  # one hint per template: the same message is one piece of evidence
        return found
