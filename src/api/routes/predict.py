from flask import Blueprint, jsonify, request

from src.api.schemas.predict_schema import validate_predict_payload
from src.api.services.prediction_service import PredictionService


predict_bp = Blueprint("predict", __name__)


@predict_bp.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    cleaned, error = validate_predict_payload(payload)
    if error is not None:
        response = {"error": error.message}
        if error.details is not None:
            response["details"] = error.details
        return jsonify(response), 400

    try:
        result = PredictionService.predict(cleaned)
    except FileNotFoundError:
        return (
            jsonify(
                {
                    "error": (
                        "Modelo de ML não encontrado. "
                        "Execute `python -m src.main --mode ml` antes de usar a API."
                    )
                }
            ),
            503,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result), 200
