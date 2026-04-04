from pathlib import Path
from typing import Any
import pandas as pd
from src.common.print_helper import PrintHelper
from src.common.plot_helper import PlotHelper
from src.pipeline.pipeline_context import PipelineContext
from src.pipeline.step import Step


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "resources" / "outputs" / "graphs" / "evaluation"


class Evaluation(Step):
    def __init__(self, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
        self.output_dir = output_dir

    def execute(self, context: PipelineContext) -> PipelineContext:
        PlotHelper.ensure_output_dir(self.output_dir)

        results = context.modeling_results
        y_test = context.data["y_test"]

        self.plot_confusion_matrices(results, y_test)

        metrics_df = self.create_metrics_dataframe(results)
        PlotHelper.plot_metrics_comparison(metrics_df, self.output_dir)
        PrintHelper.print_metrics_summary(metrics_df)

        print(f"\nGráficos de avaliação salvos em: {self.output_dir}")

        return context

    def create_metrics_dataframe(
        self,
        results: list[dict[str, Any]]
    ) -> pd.DataFrame:
        metrics_data = []

        for result in results:
            metrics_data.append(
                {
                    "model_name": result["model_name"],
                    "accuracy": result["accuracy"],
                    "recall": result["recall"],
                    "f1_score": result["f1_score"]
                }
            )

        return pd.DataFrame(metrics_data)

    def plot_confusion_matrices(
        self,
        results: list[dict[str, Any]],
        y_test: Any
    ) -> None:
        for result in results:
            PlotHelper.plot_confusion_matrix(
                y_test=y_test,
                y_pred=result["predictions"],
                model_name=result["model_name"],
                output_dir=self.output_dir
            )