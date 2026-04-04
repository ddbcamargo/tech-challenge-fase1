from pathlib import Path
from typing import Any
import pandas as pd
import shap
from src.common.plot_helper import PlotHelper
from src.pipeline.pipeline_context import PipelineContext
from src.pipeline.step import Step


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "resources" / "outputs" / "graphs" / "explainability"


class ShapAnalysis(Step):
    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR
    ) -> None:
        self.output_dir = output_dir

    def execute(self, context: PipelineContext) -> PipelineContext:
        x_df = self.convert_to_dataframe(context.data["x_train"])
        model = context.tuning_results["random_forest"]["best_model"]

        explanation = self.get_explanation(model, x_df)

        PlotHelper.plot_summary_bar(
            explanation=explanation,
            output_dir=self.output_dir
        )
        PlotHelper.plot_summary_beeswarm(
            explanation=explanation,
            output_dir=self.output_dir
        )

        print(f"\nGráficos SHAP salvos em: {self.output_dir}")

        return context

    def convert_to_dataframe(self, x_data: Any) -> pd.DataFrame:
        if isinstance(x_data, pd.DataFrame):
            return x_data

        return pd.DataFrame(x_data, columns=list(x_data.columns))

    def get_explanation(self, model: Any, x_df: pd.DataFrame) -> shap.Explanation:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_df)

        positive_class_index = list(model.classes_).index(1)

        if isinstance(shap_values, list):
            selected_values = shap_values[positive_class_index]
        elif len(shap_values.shape) == 3:
            selected_values = shap_values[:, :, positive_class_index]
        else:
            selected_values = shap_values

        return shap.Explanation(
            values=selected_values,
            data=x_df.values,
            feature_names=x_df.columns.tolist()
        )