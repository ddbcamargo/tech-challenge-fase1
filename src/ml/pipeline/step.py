from abc import ABC, abstractmethod

from src.ml.pipeline.pipeline_context import PipelineContext


class Step(ABC):
    """Contrato de uma etapa do pipeline de ML."""

    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError
