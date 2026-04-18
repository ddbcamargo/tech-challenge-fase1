from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


class PlotHelper:
    @staticmethod
    def ensure_output_dir(output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def plot_target_distribution(df: pd.DataFrame, output_dir: Path) -> None:
        PlotHelper.ensure_output_dir(output_dir)
        plt.figure(figsize=(8, 5))
        sns.countplot(x="diagnosis", data=df)
        plt.title("Distribuição do diagnóstico")
        plt.xlabel("Diagnóstico (0 = Benigno, 1 = Maligno)")
        plt.ylabel("Quantidade")
        plt.tight_layout()
        plt.savefig(output_dir / "target_distribution.png")
        plt.close()

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
        PlotHelper.ensure_output_dir(output_dir)
        plt.figure(figsize=(18, 12))
        sns.heatmap(df.corr(), cmap="coolwarm")
        plt.title("Mapa de Correlação")
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png")
        plt.close()

    @staticmethod
    def plot_confusion_matrix(
        y_test,
        y_pred,
        model_name: str,
        output_dir: Path,
    ) -> None:
        PlotHelper.ensure_output_dir(output_dir)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Benigno", "Maligno"],
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, colorbar=False)
        plt.title(f"Matriz de Confusão - {model_name}")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        )
        plt.close()

    @staticmethod
    def plot_metrics_comparison(
        metrics_df: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        PlotHelper.ensure_output_dir(output_dir)

        df_plot = metrics_df.set_index("model_name")

        ax = df_plot.plot(kind="bar", figsize=(10, 6))
        ax.set_title("Comparação de Métricas entre Modelos")
        ax.set_xlabel("Modelo")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.05)
        plt.xticks(rotation=0)
        plt.legend(["Accuracy", "Recall", "F1-Score"])
        plt.tight_layout()
        plt.savefig(output_dir / "metrics_comparison.png")
        plt.close()

    @staticmethod
    def plot_feature_importance(
        importance_df: pd.DataFrame,
        top_n: int,
        output_dir: Path,
    ) -> None:
        PlotHelper.ensure_output_dir(output_dir)

        top_features = (
            importance_df.head(top_n).sort_values(by="importance", ascending=True)
        )

        plt.figure(figsize=(10, 6))
        plt.barh(top_features["feature"], top_features["importance"])
        plt.title(f"Top {top_n} Features - Random Forest")
        plt.xlabel("Importância")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(output_dir / "random_forest_feature_importance.png")
        plt.close()

    @staticmethod
    def plot_summary_bar(explanation, output_dir: Path) -> None:
        PlotHelper.ensure_output_dir(output_dir)

        plt.figure(figsize=(14, 8))
        shap.plots.bar(explanation, max_display=10, show=False)
        plt.gcf().set_size_inches(14, 8)
        plt.savefig(
            output_dir / "shap_summary_bar.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    @staticmethod
    def plot_summary_beeswarm(explanation, output_dir: Path) -> None:
        PlotHelper.ensure_output_dir(output_dir)

        plt.figure(figsize=(14, 8))
        shap.plots.beeswarm(explanation, max_display=10, show=False)
        plt.gcf().set_size_inches(14, 8)
        plt.savefig(
            output_dir / "shap_summary_beeswarm.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
