import argparse


def run_ml() -> None:
    from src.ml.pipeline.machine_learning_pipeline import MachineLearningPipeline

    MachineLearningPipeline().run()
    print("\n===== FIM ML =====")


def run_api() -> None:
    from src.api.app import app

    print("\n===== INICIANDO API =====")
    app.run(host="0.0.0.0", port=5000, debug=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Breast Cancer ML Project")
    parser.add_argument(
        "--mode",
        choices=["ml", "api"],
        default=None,
        help=(
            "Qual fluxo executar: 'ml' (apenas treino), 'api' (apenas API) "
            "ou omita para executar ambos em sequencia (ml + api)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "ml":
        run_ml()
    elif args.mode == "api":
        run_api()
    else:
        run_ml()
        run_api()


if __name__ == "__main__":
    main()
