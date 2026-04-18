"""Spec OpenAPI 2.0 (Swagger) usada pelo Flasgger.

Mantida como dicionário Python (em vez de YAML) para reaproveitar a lista
oficial de features definida em :mod:`src.shared.constants.features`,
evitando duplicação entre código e documentação.
"""
from __future__ import annotations

from typing import Any

from src.shared.constants.features import API_TO_DATASET_FEATURE


# Valores de exemplo realistas para a UI do Swagger e testes manuais.
_EXAMPLE_PAYLOAD: dict[str, float] = {
    "radius_mean": 14.2,
    "texture_mean": 20.1,
    "perimeter_mean": 92.0,
    "area_mean": 654.0,
    "smoothness_mean": 0.09,
    "compactness_mean": 0.1,
    "concavity_mean": 0.08,
    "concave_points_mean": 0.04,
    "symmetry_mean": 0.18,
    "fractal_dimension_mean": 0.06,
    "radius_se": 0.5,
    "texture_se": 1.2,
    "perimeter_se": 3.5,
    "area_se": 40.0,
    "smoothness_se": 0.007,
    "compactness_se": 0.02,
    "concavity_se": 0.03,
    "concave_points_se": 0.01,
    "symmetry_se": 0.02,
    "fractal_dimension_se": 0.003,
    "radius_worst": 16.5,
    "texture_worst": 25.0,
    "perimeter_worst": 110.0,
    "area_worst": 880.0,
    "smoothness_worst": 0.13,
    "compactness_worst": 0.22,
    "concavity_worst": 0.25,
    "concave_points_worst": 0.12,
    "symmetry_worst": 0.28,
    "fractal_dimension_worst": 0.09,
}


def _predict_request_definition() -> dict[str, Any]:
    feature_names = list(API_TO_DATASET_FEATURE.keys())
    properties = {
        name: {
            "type": "number",
            "format": "float",
            "example": _EXAMPLE_PAYLOAD.get(name, 0.0),
        }
        for name in feature_names
    }
    return {
        "type": "object",
        "required": feature_names,
        "properties": properties,
        "example": _EXAMPLE_PAYLOAD,
    }


def _predict_response_definition() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "prediction": {
                "type": "integer",
                "enum": [0, 1],
                "description": "0 = benigno, 1 = maligno",
                "example": 1,
            },
            "label": {
                "type": "string",
                "enum": ["benigno", "maligno"],
                "example": "maligno",
            },
            "probability": {
                "type": "number",
                "format": "float",
                "description": "Probabilidade da classe predita (0..1).",
                "example": 0.93,
            },
            "model": {
                "type": "string",
                "description": "Identificador do modelo usado.",
                "example": "random_forest",
            },
        },
    }


def _error_response_definition() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "error": {"type": "string"},
            "details": {
                "type": "object",
                "properties": {
                    "missing_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "invalid_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "unknown_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
    }


def _health_response_definition() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "example": "ok"},
            "model": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "example": "random_forest"},
                    "name": {"type": "string", "example": "Random Forest"},
                },
            },
            "message": {"type": "string"},
        },
    }


def build_openapi_spec() -> dict[str, Any]:
    return {
        "swagger": "2.0",
        "info": {
            "title": "Breast Cancer Classifier API",
            "description": (
                "API Flask que serve o melhor modelo de Machine Learning "
                "treinado para classificar tumores como benignos ou malignos."
            ),
            "version": "1.0.0",
            "contact": {
                "name": "Diego Diondré Bueno de Camargo",
                "email": "diego.diondre@gmail.com",
            },
        },
        "basePath": "/",
        "schemes": ["http"],
        "consumes": ["application/json"],
        "produces": ["application/json"],
        "tags": [
            {"name": "health", "description": "Status do serviço."},
            {"name": "prediction", "description": "Endpoints de inferência."},
        ],
        "paths": {
            "/health": {
                "get": {
                    "tags": ["health"],
                    "summary": "Health check do serviço",
                    "description": (
                        "Retorna `ok` quando o modelo está carregado e "
                        "`degraded` quando o modelo ainda não foi treinado."
                    ),
                    "responses": {
                        "200": {
                            "description": "Status do serviço.",
                            "schema": {"$ref": "#/definitions/HealthResponse"},
                        }
                    },
                }
            },
            "/predict": {
                "post": {
                    "tags": ["prediction"],
                    "summary": "Prediz diagnóstico (benigno/maligno)",
                    "description": (
                        "Recebe as 30 features clínicas do dataset Breast "
                        "Cancer e retorna a classe predita, a probabilidade "
                        "e o identificador do modelo usado.\n\n"
                        "Aceita tanto nomes em snake_case "
                        "(`concave_points_mean`) quanto os nomes originais "
                        "do dataset (`concave points_mean`)."
                    ),
                    "parameters": [
                        {
                            "in": "body",
                            "name": "body",
                            "required": True,
                            "schema": {"$ref": "#/definitions/PredictRequest"},
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Predição realizada com sucesso.",
                            "schema": {"$ref": "#/definitions/PredictResponse"},
                        },
                        "400": {
                            "description": (
                                "Payload inválido: features faltando, "
                                "tipos incorretos ou campos desconhecidos."
                            ),
                            "schema": {"$ref": "#/definitions/ErrorResponse"},
                        },
                        "503": {
                            "description": (
                                "Modelo de ML ainda não foi treinado. "
                                "Execute `python -m src.main --mode ml`."
                            ),
                            "schema": {"$ref": "#/definitions/ErrorResponse"},
                        },
                    },
                }
            },
        },
        "definitions": {
            "PredictRequest": _predict_request_definition(),
            "PredictResponse": _predict_response_definition(),
            "ErrorResponse": _error_response_definition(),
            "HealthResponse": _health_response_definition(),
        },
    }


# Configuração do Flasgger: monta a UI em /apidocs/ e o JSON em /apispec.json.
SWAGGER_CONFIG: dict[str, Any] = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/",
}
