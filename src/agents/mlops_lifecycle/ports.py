from typing import Protocol

from src.contracts.payloads import ModelHealthAssessed
from src.contracts.tasks import TaskContext


class ModelHealthAssessor(Protocol):
    async def assess(self, context: TaskContext) -> ModelHealthAssessed: ...


class TrainingPipeline(Protocol):
    async def train_candidate(self, dataset_ref: str, pipeline_version: str) -> str: ...


class ModelRegistry(Protocol):
    async def register_candidate(self, run_id: str, model_name: str) -> str: ...
