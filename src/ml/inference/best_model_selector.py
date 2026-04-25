"""Seleção do melhor modelo a partir dos resultados do pipeline.

Estratégia:
- Ordena os modelos por (recall, f1_score, accuracy) em ordem decrescente.
- O modelo com maior recall vence; em caso de empate, decide-se por f1_score
  e, por fim, accuracy. Isso reflete a prioridade do domínio médico (reduzir
  falsos negativos).
"""
from __future__ import annotations

from typing import Any


def select_best_model(
    modeling_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Retorna o dicionário de métricas do modelo vencedor.

    O dict já contém `model_name`, `model_key`, `accuracy`, `recall`,
    `f1_score` e `requires_scaler`.
    """
    if not modeling_results:
        raise ValueError("Nenhum resultado de modelagem disponível para seleção.")

    ranked = sorted(
        modeling_results,
        key=lambda r: (
            float(r["recall"]),
            float(r["f1_score"]),
            float(r["accuracy"]),
        ),
        reverse=True,
    )

    return ranked[0]
