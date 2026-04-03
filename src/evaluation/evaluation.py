from pathlib import Path
import os
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent.parent.parent
GRAPHS_PATH = BASE_DIR / "resources" / "outputs" / "graphs" / "evaluation"


def ensure_output_dir(output_dir: Path = GRAPHS_PATH) -> None:
    os.makedirs(output_dir, exist_ok=True)


def plot_confusion_matrix(
    y_test: Any,
    y_pred: Any,
    model_name: str,
    output_dir: Path = GRAPHS_PATH
) -> None:
    ensure_output_dir(output_dir)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benigno", "Maligno"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"))
    plt.close()


def create_metrics_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    metrics_data = []

    for result in results:
        metrics_data.append({
            "model_name": result["model_name"],
            "accuracy": result["accuracy"],
            "recall": result["recall"],
            "f1_score": result["f1_score"]
        })

    return pd.DataFrame(metrics_data)


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    output_dir: Path = GRAPHS_PATH
) -> None:
    ensure_output_dir(output_dir)

    df_plot = metrics_df.set_index("model_name")

    plt.figure(figsize=(10, 6))
    df_plot.plot(kind="bar", figsize=(10, 6))
    plt.title("Comparação de Métricas entre Modelos")
    plt.xlabel("Modelo")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.05)
    plt.xticks(rotation=0)
    plt.legend(["Accuracy", "Recall", "F1-Score"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"))
    plt.close()


def print_metrics_summary(metrics_df: pd.DataFrame) -> None:
    print("\n=== MODEL COMPARISON ===")
    print(metrics_df.sort_values(by=["recall", "f1_score", "accuracy"], ascending=False).to_string(index=False))


def run_evaluation(
    results: List[Dict[str, Any]],
    y_test: Any,
    output_dir: Path = GRAPHS_PATH
) -> pd.DataFrame:
    ensure_output_dir(output_dir)

    for result in results:
        plot_confusion_matrix(
            y_test=y_test,
            y_pred=result["predictions"],
            model_name=result["model_name"],
            output_dir=output_dir
        )

    metrics_df = create_metrics_dataframe(results)
    plot_metrics_comparison(metrics_df, output_dir)
    print_metrics_summary(metrics_df)

    print(f"\nGráficos de avaliação salvos em: {output_dir}")

    return metrics_df