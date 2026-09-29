from typing import Protocol

from src.contracts.payloads import ResourceRecommendation
from src.contracts.tasks import TaskContext


class ResourcePlanner(Protocol):
    async def recommend(self, context: TaskContext) -> ResourceRecommendation | None: ...
