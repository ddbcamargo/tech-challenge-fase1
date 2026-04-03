from pathlib import Path
from typing import Any
import pandas as pd
import shap
from src.common.plot_helper import PlotHelper


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "resources" / "outputs" / "graphs" / "explainability"


class ShapAnalysis:
    def __init__(
        self,
        model: Any,
        x_data: Any,
        feature_names: list[str],
        output_dir: Path = DEFAULT_OUTPUT_DIR
    ) -> None:
        self.model = model
        self.x_data = x_data
        self.feature_names = feature_names
        self.output_dir = output_dir

    def convert_to_dataframe(self) -> pd.DataFrame:
        if isinstance(self.x_data, pd.DataFrame):
            return self.x_data

        return pd.DataFrame(self.x_data, columns=self.feature_names)

    def get_explanation(self, x_df: pd.DataFrame):
        explainer = shap.Explainer(self.model, x_df)
        explanation = explainer(x_df)

        if len(explanation.values.shape) == 3:
            explanation = explanation[:, :, 1]

        return explanation

    def run(self):
        x_df = self.convert_to_dataframe()
        explanation = self.get_explanation(x_df)

        PlotHelper.plot_summary_bar(
            explanation=explanation,
            output_dir=self.output_dir
        )
        PlotHelper.plot_summary_beeswarm(
            explanation=explanation,
            output_dir=self.output_dir
        )

        print(f"\nGráficos SHAP salvos em: {self.output_dir}")

        return explanation


def run_shap_analysis(
    model: Any,
    x_data: Any,
    feature_names: list[str]
):
    shap_analysis = ShapAnalysis(
        model=model,
        x_data=x_data,
        feature_names=feature_names
    )
    return shap_analysis.run()