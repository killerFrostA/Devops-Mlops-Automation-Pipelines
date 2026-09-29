from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from src.contracts.base import Contract
from src.contracts.enums import Environment, Priority, Severity
from src.contracts.topics import TOPICS


class ServiceIdentity(Contract):
    service_id: Annotated[str, Field(min_length=1)]
    namespace: Annotated[str, Field(min_length=1)]
    version: Annotated[str, Field(min_length=1)]


class EventEnvelope(Contract):
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    schema_version: Literal["1.0"] = "1.0"
    correlation_id: Annotated[str, Field(min_length=1)]
    causation_id: str | None = None
    trace_id: Annotated[str, Field(min_length=1)]
    source: Annotated[str, Field(min_length=1)]
    target: Annotated[str, Field(min_length=1)]
    topic: str
    message_type: str
    priority: Priority
    severity: Severity
    timestamp: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    deadline: AwareDatetime | None = None
    environment: Environment
    service: ServiceIdentity
    evidence_refs: tuple[str, ...] = ()
    confidence: Annotated[float | None, Field(ge=0, le=1, allow_inf_nan=False)] = None
    requires_ack: bool = True
    idempotency_key: Annotated[str, Field(min_length=1)]
    payload: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_event_semantics(self) -> "EventEnvelope":
        spec = TOPICS.get(self.topic)
        if spec is None:
            raise ValueError(f"Unsupported v1 topic: {self.topic}")
        message_type, payload_model = spec
        if self.message_type != message_type:
            raise ValueError(f"Topic {self.topic} requires message_type={message_type}")
        payload_model.model_validate(self.payload)
        if self.deadline is not None and self.deadline <= self.timestamp:
            raise ValueError("deadline must be later than timestamp")
        return self

    @property
    def partition_key(self) -> str:
        """Incident streams preserve order only within this Kafka partition key."""
        return self.correlation_id
