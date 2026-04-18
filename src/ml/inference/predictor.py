"""Inferência do melhor modelo de ML salvo em disco.

A classe :class:`MLPredictor` encapsula o carregamento dos artefatos
(``best_model.joblib``, ``scaler.joblib``, ``best_model_info.json``) e
expõe um método :meth:`predict` que aceita um dicionário de features e
retorna o resultado pronto para a API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.inference.persist_best_model import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    SCALER_FILENAME,
)
from src.shared.artifacts.storage import load_joblib, load_json
from src.shared.config.paths import ML_MODELS_DIR
from src.shared.constants.features import (
    API_TO_DATASET_FEATURE,
    CLASS_LABELS,
)


class MLPredictor:
    def __init__(self, models_dir: Path = ML_MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.metadata: dict[str, Any] = load_json(models_dir / METADATA_FILENAME)
        self.model = load_joblib(models_dir / MODEL_FILENAME)
        self.scaler = load_joblib(models_dir / SCALER_FILENAME)
        self.feature_names: list[str] = self.metadata["feature_names"]
        self.requires_scaler: bool = self.metadata.get("requires_scaler", False)
        self.model_key: str = self.metadata["model_key"]
        self.model_name: str = self.metadata["model_name"]

    # ------------------------------------------------------------------
    # Helpers públicos
    # ------------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "requires_scaler": self.requires_scaler,
            "metrics": self.metadata.get("metrics", {}),
            "feature_names": self.feature_names,
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Executa a predição a partir do payload JSON recebido pela API.

        Aceita tanto nomes com espaço (``concave points_mean``) quanto
        snake_case (``concave_points_mean``).
        """
        features_df = self._build_feature_frame(payload)
        x = self._apply_scaler(features_df)

        prediction = int(self.model.predict(x)[0])
        probability = self._predict_probability(x, prediction)

        return {
            "prediction": prediction,
            "label": CLASS_LABELS[prediction],
            "probability": probability,
            "model": self.model_key,
        }

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------
    def _build_feature_frame(self, payload: dict[str, Any]) -> pd.DataFrame:
        row: dict[str, float] = {}
        missing: list[str] = []

        for dataset_name in self.feature_names:
            api_name = dataset_name.replace(" ", "_")
            if dataset_name in payload:
                row[dataset_name] = float(payload[dataset_name])
            elif api_name in payload:
                row[dataset_name] = float(payload[api_name])
            else:
                missing.append(api_name)

        if missing:
            raise ValueError(f"Features ausentes no payload: {missing}")

        return pd.DataFrame([[row[name] for name in self.feature_names]],
                            columns=self.feature_names)

    def _apply_scaler(self, features_df: pd.DataFrame) -> np.ndarray | pd.DataFrame:
        if self.requires_scaler:
            return self.scaler.transform(features_df)
        return features_df

    def _predict_probability(
        self,
        x: Any,
        prediction: int,
    ) -> float | None:
        if not hasattr(self.model, "predict_proba"):
            return None
        proba = self.model.predict_proba(x)[0]
        return float(proba[prediction])


# Mapeamento reexportado por conveniência (evita import duplo na API).
API_FEATURE_ALIASES = API_TO_DATASET_FEATURE
