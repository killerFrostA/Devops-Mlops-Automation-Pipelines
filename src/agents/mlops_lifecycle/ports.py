from typing import Protocol

from src.contracts.mlops import ModelHealthInput, ModelHealthReport


class ModelHealthAssessor(Protocol):
    async def assess(self, request: ModelHealthInput) -> ModelHealthReport: ...


class TrainingPipeline(Protocol):
    async def train_candidate(self, dataset_ref: str, pipeline_version: str) -> str: ...


class ModelRegistry(Protocol):
    async def register_candidate(self, run_id: str, model_name: str) -> str: ...
