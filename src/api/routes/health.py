from flask import Blueprint, jsonify

from src.api.services.prediction_service import PredictionService


health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health_check():
    payload = {"status": "ok"}
    try:
        info = PredictionService.info()
        payload["model"] = {
            "key": info["model_key"],
            "name": info["model_name"],
        }
    except FileNotFoundError:
        payload["status"] = "degraded"
        payload["model"] = None
        payload["message"] = (
            "Modelo de ML ainda não treinado. Execute `python -m src.main --mode ml`."
        )
    return jsonify(payload), 200
