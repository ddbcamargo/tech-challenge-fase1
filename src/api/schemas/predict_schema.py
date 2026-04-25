from dataclasses import dataclass
from typing import Any

from src.shared.constants.features import API_TO_DATASET_FEATURE


@dataclass
class ValidationError:
    message: str
    details: dict[str, Any] | None = None


def validate_predict_payload(
    payload: Any,
) -> tuple[dict[str, float] | None, ValidationError | None]:
    """Valida o corpo da requisição.

    Regras:
    - Deve ser um objeto JSON (dict).
    - Todas as features esperadas devem estar presentes.
    - Todos os valores devem ser numéricos.
    """
    if not isinstance(payload, dict):
        return None, ValidationError(
            message="O corpo da requisição deve ser um objeto JSON.",
        )

    allowed_keys = set(API_TO_DATASET_FEATURE.keys()) | set(
        API_TO_DATASET_FEATURE.values()
    )
    unknown_keys = [k for k in payload.keys() if k not in allowed_keys]

    missing: list[str] = []
    invalid: list[str] = []
    cleaned: dict[str, float] = {}

    for api_name, dataset_name in API_TO_DATASET_FEATURE.items():
        if api_name in payload:
            raw = payload[api_name]
        elif dataset_name in payload:
            raw = payload[dataset_name]
        else:
            missing.append(api_name)
            continue

        try:
            cleaned[dataset_name] = float(raw)
        except (TypeError, ValueError):
            invalid.append(api_name)

    if missing or invalid or unknown_keys:
        return None, ValidationError(
            message="Payload inválido.",
            details={
                "missing_fields": missing,
                "invalid_fields": invalid,
                "unknown_fields": unknown_keys,
            },
        )

    return cleaned, None
