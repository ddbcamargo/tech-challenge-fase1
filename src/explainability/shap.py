import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import shap


def ensure_output_dir(output_dir: str = "../resources/outputs/graphs") -> None:
    os.makedirs(output_dir, exist_ok=True)


def convert_to_dataframe(x_data: Any, feature_names: list[str]) -> pd.DataFrame:
    if isinstance(x_data, pd.DataFrame):
        return x_data

    return pd.DataFrame(x_data, columns=feature_names)


def get_shap_explanation(model: Any, x_data: pd.DataFrame):
    explainer = shap.Explainer(model, x_data)
    explanation = explainer(x_data)

    # Classificação binária: mantém só a classe positiva
    if len(explanation.values.shape) == 3:
        explanation = explanation[:, :, 1]

    return explanation


def plot_shap_summary_bar(
    explanation,
    output_dir: str = "../resources/outputs/graphs"
) -> None:
    ensure_output_dir(output_dir)

    plt.figure(figsize=(14, 8))
    shap.plots.bar(explanation, max_display=10, show=False)
    plt.gcf().set_size_inches(14, 8)
    plt.savefig(
        os.path.join(output_dir, "shap_summary_bar.png"),
        bbox_inches="tight",
        dpi=300
    )
    plt.close()


def plot_shap_summary_beeswarm(
    explanation,
    output_dir: str = "../resources/outputs/graphs"
) -> None:
    ensure_output_dir(output_dir)

    plt.figure(figsize=(14, 8))
    shap.plots.beeswarm(explanation, max_display=10, show=False)
    plt.gcf().set_size_inches(14, 8)
    plt.savefig(
        os.path.join(output_dir, "shap_summary_beeswarm.png"),
        bbox_inches="tight",
        dpi=300
    )
    plt.close()


def run_shap(
    model: Any,
    x_data: Any,
    feature_names: list[str],
    output_dir: str = "../resources/outputs/graphs"
):
    x_df = convert_to_dataframe(x_data, feature_names)
    explanation = get_shap_explanation(model, x_df)

    plot_shap_summary_bar(explanation, output_dir)
    plot_shap_summary_beeswarm(explanation, output_dir)

    print(f"\nGráficos SHAP salvos em: {output_dir}")

    return explanation