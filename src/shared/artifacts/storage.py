"""Funções utilitárias para salvar e carregar artefatos (modelos, scalers, metadados).

Mantém a persistência de ML centralizada para que tanto o pipeline de treino
quanto a API de inferência usem o mesmo contrato.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_joblib(obj: Any, path: Path) -> Path:
    """Salva um objeto arbitrário usando joblib (pickle otimizado)."""
    ensure_dir(path.parent)
    joblib.dump(obj, path)
    return path


def load_joblib(path: Path) -> Any:
    """Carrega um objeto salvo via :func:`save_joblib`."""
    if not path.exists():
        raise FileNotFoundError(f"Artefato não encontrado em: {path}")
    return joblib.load(path)


def save_json(data: dict[str, Any], path: Path) -> Path:
    """Salva um dicionário como JSON (usado para metadados do modelo)."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metadado não encontrado em: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
