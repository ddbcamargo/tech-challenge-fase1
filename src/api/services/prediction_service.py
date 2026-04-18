"""Serviço de inferência consumido pelas rotas da API.

Implementa cache lazy do ``MLPredictor`` para que o modelo e o scaler
sejam carregados apenas uma vez por processo.
"""
from __future__ import annotations

from typing import Any

from src.ml.inference.predictor import MLPredictor


class PredictionService:
    _predictor: MLPredictor | None = None

    @classmethod
    def get_predictor(cls) -> MLPredictor:
        if cls._predictor is None:
            cls._predictor = MLPredictor()
        return cls._predictor

    @classmethod
    def predict(cls, payload: dict[str, Any]) -> dict[str, Any]:
        return cls.get_predictor().predict(payload)

    @classmethod
    def info(cls) -> dict[str, Any]:
        return cls.get_predictor().info()

    @classmethod
    def reset(cls) -> None:
        """Reset do cache (útil em testes)."""
        cls._predictor = None
