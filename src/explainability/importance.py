from pathlib import Path
import os
from typing import Any
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent
GRAPHS_PATH = BASE_DIR / "resources" / "outputs" / "graphs" / "explainability"


def ensure_output_dir(output_dir: Path = GRAPHS_PATH) -> None:
    os.makedirs(output_dir, exist_ok=True)


def create_feature_importance_dataframe(model: Any, feature_names: list[str]) -> pd.DataFrame:
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(by="importance", ascending=False).reset_index(drop=True)

    return importance_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 10,
    output_dir: Path = GRAPHS_PATH
) -> None:
    ensure_output_dir(output_dir)

    top_features = importance_df.head(top_n).sort_values(by="importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title(f"Top {top_n} Features - Random Forest")
    plt.xlabel("Importância")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "random_forest_feature_importance.png"))
    plt.close()


def print_feature_importance(importance_df: pd.DataFrame, top_n: int = 10) -> None:
    print(f"\n=== TOP {top_n} FEATURE IMPORTANCE - RANDOM FOREST ===")
    print(importance_df.head(top_n).to_string(index=False))


def run_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 10,
    output_dir: Path = GRAPHS_PATH
) -> pd.DataFrame:
    importance_df = create_feature_importance_dataframe(model, feature_names)
    print_feature_importance(importance_df, top_n)
    plot_feature_importance(importance_df, top_n, output_dir)

    print(f"\nGráfico de feature importance salvo em: {output_dir}")

    return importance_df