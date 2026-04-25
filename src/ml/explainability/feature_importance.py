from pathlib import Path
from typing import Any

import pandas as pd

from src.ml.pipeline.pipeline_context import PipelineContext
from src.ml.pipeline.step import Step
from src.shared.config.paths import EXPLAINABILITY_GRAPHS_DIR
from src.shared.utils.plot_helper import PlotHelper
from src.shared.utils.print_helper import PrintHelper


class FeatureImportance(Step):
    def __init__(
        self,
        top_n: int = 10,
        output_dir: Path = EXPLAINABILITY_GRAPHS_DIR,
    ) -> None:
        self.top_n = top_n
        self.output_dir = output_dir

    def execute(self, context: PipelineContext) -> PipelineContext:
        model = context.tuning_results["random_forest"]["best_model"]
        feature_names = list(context.data["x_train"].columns)
        importance_df = self.create_feature_importance_dataframe(
            model=model,
            feature_names=feature_names,
        )

        PrintHelper.print_feature_importance(
            importance_df=importance_df,
            top_n=self.top_n,
        )

        PlotHelper.plot_feature_importance(
            importance_df=importance_df,
            top_n=self.top_n,
            output_dir=self.output_dir,
        )

        print(f"\nGráfico de feature importance salvo em: {self.output_dir}")

        return context

    def create_feature_importance_dataframe(
        self,
        model: Any,
        feature_names: list[str],
    ) -> pd.DataFrame:
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )

        return (
            importance_df.sort_values(by="importance", ascending=False).reset_index(
                drop=True
            )
        )
