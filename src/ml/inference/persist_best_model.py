"""Etapa final do pipeline: persiste o melhor modelo + scaler + metadados.

Artefatos gerados em ``models/ml/``:
    - ``best_model.joblib``            → modelo vencedor (sklearn)
    - ``scaler.joblib``                → StandardScaler treinado
    - ``best_model_info.json``         → metadados (nome, métricas, features)
"""
from __future__ import annotations

from pathlib import Path

from src.ml.inference.best_model_selector import select_best_model
from src.ml.pipeline.pipeline_context import PipelineContext
from src.ml.pipeline.step import Step
from src.shared.artifacts.storage import save_joblib, save_json
from src.shared.config.paths import ML_MODELS_DIR


MODEL_FILENAME = "best_model.joblib"
SCALER_FILENAME = "scaler.joblib"
METADATA_FILENAME = "best_model_info.json"


class PersistBestModel(Step):
    def __init__(self, output_dir: Path = ML_MODELS_DIR) -> None:
        self.output_dir = output_dir

    def execute(self, context: PipelineContext) -> PipelineContext:
        best = select_best_model(context.modeling_results)
        model_key: str = best["model_key"]

        best_model = context.tuning_results[model_key]["best_model"]
        scaler = context.data["scaler"]
        feature_names = context.data["feature_names"]

        model_path = self.output_dir / MODEL_FILENAME
        scaler_path = self.output_dir / SCALER_FILENAME
        metadata_path = self.output_dir / METADATA_FILENAME

        save_joblib(best_model, model_path)
        save_joblib(scaler, scaler_path)

        metadata = {
            "model_key": model_key,
            "model_name": best["model_name"],
            "requires_scaler": bool(best["requires_scaler"]),
            "metrics": {
                "accuracy": float(best["accuracy"]),
                "recall": float(best["recall"]),
                "f1_score": float(best["f1_score"]),
            },
            "feature_names": feature_names,
            "artifacts": {
                "model": MODEL_FILENAME,
                "scaler": SCALER_FILENAME,
            },
        }
        save_json(metadata, metadata_path)

        context.best_model_info = metadata

        print("\n=== BEST MODEL PERSISTED ===")
        print(f"Melhor modelo: {metadata['model_name']} ({model_key})")
        print(f"  recall   = {metadata['metrics']['recall']:.4f}")
        print(f"  f1_score = {metadata['metrics']['f1_score']:.4f}")
        print(f"  accuracy = {metadata['metrics']['accuracy']:.4f}")
        print(f"Artefatos salvos em: {self.output_dir}")

        return context
