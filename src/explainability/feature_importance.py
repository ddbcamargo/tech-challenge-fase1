from pathlib import Path
from typing import Any
import pandas as pd
from src.common.print_helper import PrintHelper
from src.common.plot_helper import PlotHelper
from src.pipeline.pipeline_context import PipelineContext
from src.pipeline.step import Step


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "resources" / "outputs" / "graphs" / "explainability"


class FeatureImportance(Step):
    def __init__(
        self,
        top_n: int = 10,
        output_dir: Path = DEFAULT_OUTPUT_DIR
    ) -> None:
        self.top_n = top_n
        self.output_dir = output_dir

    def execute(self, context: PipelineContext) -> PipelineContext:
        model = context.tuning_results["random_forest"]["best_model"] ##TODO: Identificar melhor modelo automaticamente
        importance_df = self.create_feature_importance_dataframe(model)

        PrintHelper.print_feature_importance(
            importance_df=importance_df,
            top_n=self.top_n
        )

        PlotHelper.plot_feature_importance(
            importance_df=importance_df,
            top_n=self.top_n,
            output_dir=self.output_dir
        )

        print(f"\nGráfico de feature importance salvo em: {self.output_dir}")

        return context

    def create_feature_importance_dataframe(self, model: Any) -> pd.DataFrame:
        importance_df = pd.DataFrame(
            {
                "feature": list(model.data["x_train"].columns),
                "importance": model.feature_importances_
            }
        )

        return (
            importance_df
                .sort_values(by="importance", ascending=False)
                .reset_index(drop=True)
        )