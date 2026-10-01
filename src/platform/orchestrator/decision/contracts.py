"""Auditable decision records; model advice is never execution authority."""

import re
from enum import StrEnum
from typing import Annotated

from pydantic import Field, model_validator

from src.contracts.base import Contract
from src.contracts.enums import ActionType, AgentId, Environment


class DecisionDisposition(StrEnum):
    ROUTE_ANALYSIS = "ROUTE_ANALYSIS"
    WAIT_FOR_EVIDENCE = "WAIT_FOR_EVIDENCE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    BLOCKED = "BLOCKED"
    NO_ACTION = "NO_ACTION"


class AdvisorStatus(StrEnum):
    NOT_USED = "NOT_USED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    UNAVAILABLE = "UNAVAILABLE"


class CandidateSummary(Contract):
    action_id: Annotated[str, Field(min_length=1)]
    action_type: ActionType
    action_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class HypothesisSummary(Contract):
    code: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")]
    probability: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class AdvisorRequest(Contract):
    incident_id: Annotated[str, Field(min_length=1)]
    event_id: Annotated[str, Field(min_length=1)]
    environment: Environment
    candidates: tuple[CandidateSummary, ...]
    hypotheses: Annotated[tuple[HypothesisSummary, ...], Field(min_length=1, max_length=5)]
    evidence_count: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def require_choices(self) -> "AdvisorRequest":
        if not 2 <= len(self.candidates) <= 16:
            raise ValueError("Advisor requires between two and sixteen candidates")
        if any(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", candidate.action_id) is None
            for candidate in self.candidates
        ):
            raise ValueError("Advisor candidate action IDs must be canonical")
        return self


class AdvisorSuggestion(Contract):
    candidate_action_id: Annotated[str, Field(min_length=1)]
    reason_code: Annotated[str, Field(min_length=1, max_length=80)]


class DecisionPlan(Contract):
    incident_id: Annotated[str, Field(min_length=1)]
    event_id: Annotated[str, Field(min_length=1)]
    topic: Annotated[str, Field(min_length=1)]
    disposition: DecisionDisposition
    next_agents: tuple[AgentId, ...] = ()
    candidates: tuple[CandidateSummary, ...] = ()
    reason_codes: tuple[str, ...]
    advisor_status: AdvisorStatus = AdvisorStatus.NOT_USED
    advisor_preferred_candidate_id: str | None = None

    @model_validator(mode="after")
    def check_shape(self) -> "DecisionPlan":
        if self.disposition == DecisionDisposition.ROUTE_ANALYSIS:
            if not self.next_agents:
                raise ValueError("Analysis decision requires target agents")
        elif self.next_agents:
            raise ValueError("Non-routing decision must not dispatch agents")
        if AgentId.DEPLOYMENT_RECOVERY in self.next_agents:
            raise ValueError("Decision plan cannot dispatch an action owner")
        if len(set(self.next_agents)) != len(self.next_agents):
            raise ValueError("Decision plan repeats an agent")
        ids = [candidate.action_id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("Decision plan repeats an action candidate")
        if self.advisor_preferred_candidate_id is not None:
            if (
                self.advisor_status != AdvisorStatus.ACCEPTED
                or self.advisor_preferred_candidate_id not in ids
            ):
                raise ValueError("Advisor preference must name a validated candidate")
        elif self.advisor_status == AdvisorStatus.ACCEPTED:
            raise ValueError("Accepted advisor response needs a candidate preference")
        if not self.reason_codes:
            raise ValueError("Decision requires at least one machine-readable reason")
        return self
