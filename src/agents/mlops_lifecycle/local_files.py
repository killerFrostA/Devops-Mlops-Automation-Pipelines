"""Local JSON readers for testing the Agent 6 source-backed path.

These readers are development fixtures, not production database adapters.
"""

from datetime import datetime
from hashlib import sha256
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from src.agents.mlops_lifecycle.ports import (
    BaselineRepository,
    LabelRepository,
    PredictionRepository,
)
from src.contracts.base import Contract
from src.contracts.mlops_observations import (
    BaselineProfile,
    LabelBatch,
    LabelObservation,
    ModelWindowQuery,
    PredictionBatch,
    PredictionObservation,
)

_PREDICTIONS = TypeAdapter(tuple[PredictionObservation, ...])
_LABELS = TypeAdapter(tuple[LabelObservation, ...])


class LocalSourceConfig(Contract):
    """File paths are resolved relative to the config file by the CLI."""

    query: ModelWindowQuery
    baseline_file: Path
    predictions_file: Path
    labels_file: Path


def _evidence_ref(contents: bytes) -> str:
    """Identify the exact source bytes used in a local assessment."""
    return f"sha256:{sha256(contents).hexdigest()}"


class FileBaselineRepository(BaselineRepository):
    def __init__(self, path: Path) -> None:
        self.path = path

    async def get(self, model_name: str, model_version: str) -> BaselineProfile:
        contents = self.path.read_bytes()
        try:
            baseline = BaselineProfile.model_validate_json(contents)
        except ValidationError as error:
            raise ValueError("Invalid baseline JSON") from error
        if (baseline.model_name, baseline.model_version) != (model_name, model_version):
            raise ValueError("Baseline model identity differs from the query")
        # The file digest is provenance for this local run; no evidence store exists yet.
        return baseline.model_copy(update={"evidence_ref": _evidence_ref(contents)})


class FilePredictionRepository(PredictionRepository):
    def __init__(self, path: Path) -> None:
        self.path = path

    async def list_window(
        self, model_name: str, model_version: str, window_start: datetime, window_end: datetime
    ) -> PredictionBatch:
        contents = self.path.read_bytes()
        try:
            records = _PREDICTIONS.validate_json(contents)
        except ValidationError as error:
            raise ValueError("Invalid prediction-record JSON") from error
        selected = tuple(
            record
            for record in records
            if (record.model_name, record.model_version) == (model_name, model_version)
            and window_start <= record.predicted_at < window_end
        )
        return PredictionBatch(records=selected, evidence_ref=_evidence_ref(contents))


class FileLabelRepository(LabelRepository):
    def __init__(self, path: Path) -> None:
        self.path = path

    async def list_confirmed(self, observation_ids: tuple[str, ...], as_of: datetime) -> LabelBatch:
        contents = self.path.read_bytes()
        try:
            records = _LABELS.validate_json(contents)
        except ValidationError as error:
            raise ValueError("Invalid confirmed-label JSON") from error
        wanted = set(observation_ids)
        selected = tuple(
            record
            for record in records
            if record.observation_id in wanted and record.confirmed_at <= as_of
        )
        return LabelBatch(records=selected, evidence_ref=_evidence_ref(contents))
