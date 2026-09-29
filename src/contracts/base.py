from pydantic import BaseModel, ConfigDict


class Contract(BaseModel):
    """Reject accidental contract drift; treat records as immutable values."""

    model_config = ConfigDict(extra="forbid", frozen=True)
