from datetime import datetime
from typing import Protocol

from src.contracts.mlops import ModelHealthInput, ModelHealthReport
from src.contracts.mlops_observations import BaselineProfile, LabelBatch, PredictionBatch


class ModelHealthAssessor(Protocol):
    async def assess(self, request: ModelHealthInput) -> ModelHealthReport: ...


class BaselineRepository(Protocol):
    async def get(self, model_name: str, model_version: str) -> BaselineProfile: ...


class PredictionRepository(Protocol):
    async def list_window(
        self, model_name: str, model_version: str, window_start: datetime, window_end: datetime
    ) -> PredictionBatch: ...


class LabelRepository(Protocol):
    async def list_confirmed(
        self, observation_ids: tuple[str, ...], as_of: datetime
    ) -> LabelBatch: ...


class TrainingPipeline(Protocol):
    async def train_candidate(self, dataset_ref: str, pipeline_version: str) -> str: ...


class ModelRegistry(Protocol):
    async def register_candidate(self, run_id: str, model_name: str) -> str: ...
