from typing import Annotated

from pydantic import AwareDatetime, Field

from src.contracts.base import Contract


class RetryPolicy(Contract):
    max_attempts: Annotated[int, Field(ge=1)] = 3
    initial_backoff_seconds: Annotated[float, Field(gt=0)] = 1.0
    max_backoff_seconds: Annotated[float, Field(gt=0)] = 30.0
    # Worker implementation must add jitter, deadlines and retry-safe error classification.


class DeadLetterRecord(Contract):
    original_topic: str
    original_message_id: str
    original_payload_ref: str
    attempt: Annotated[int, Field(ge=1)]
    error_code: str
    retryable: bool
    first_failed_at: AwareDatetime
    last_failed_at: AwareDatetime
