from src.pipeline.pipeline_context import PipelineContext
from abc import ABC, abstractmethod

class Step(ABC):
    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError