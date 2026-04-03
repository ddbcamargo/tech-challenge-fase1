from pathlib import Path
from typing import Any
import pandas as pd
from src.common.print_helper import PrintHelper
from src.common.plot_helper import PlotHelper


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "resources" / "outputs" / "graphs" / "explainability"


class FeatureImportance:
    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        top_n: int = 10,
        output_dir: Path = DEFAULT_OUTPUT_DIR
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.top_n = top_n
        self.output_dir = output_dir

    def create_feature_importance_dataframe(self) -> pd.DataFrame:
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_
            }
        )

        return (
            importance_df
            .sort_values(by="importance", ascending=False)
            .reset_index(drop=True)
        )

    def run(self) -> pd.DataFrame:
        importance_df = self.create_feature_importance_dataframe()

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

        return importance_df


def run_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 10
) -> pd.DataFrame:
    importance = FeatureImportance(
        model=model,
        feature_names=feature_names,
        top_n=top_n
    )

    return importance.run()