"""Entrypoint do projeto.

Uso:
    python -m src.main              # treina o pipeline de ML (padrao)
    python -m src.main --mode ml    # pipeline de Machine Learning
    python -m src.main --mode api   # sobe a API Flask para inferencia
"""
from __future__ import annotations

import argparse


def run_ml() -> None:
    from src.ml.pipeline.machine_learning_pipeline import MachineLearningPipeline

    MachineLearningPipeline().run()
    print("\n===== FIM ML =====")


def run_api() -> None:
    from src.api.app import app

    app.run(host="0.0.0.0", port=5000, debug=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Breast Cancer ML Project")
    parser.add_argument(
        "--mode",
        choices=["ml", "api"],
        default="ml",
        help="Qual fluxo executar: treino (ml) ou API (api).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "ml":
        run_ml()
    elif args.mode == "api":
        run_api()


if __name__ == "__main__":
    main()
