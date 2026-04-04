from pathlib import Path
import pandas as pd
from src.common.print_helper import PrintHelper
from src.common.plot_helper import PlotHelper
from src.pipeline.pipeline_context import PipelineContext
from src.pipeline.step import Step

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "resources" / "data" / "breast_cancer.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "resources" / "outputs" / "graphs" / "eda"


class ExploratoryDataAnalysis(Step):
    def __init__(
        self,
        data_path: Path = DEFAULT_DATA_PATH,
        output_dir: Path = DEFAULT_OUTPUT_DIR
    ) -> None:
        self.data_path = data_path
        self.output_dir = output_dir


    def execute(self, context: PipelineContext) -> PipelineContext:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        context.df = self.load_data()
        context.df = self.clean_data(context.df)

        PrintHelper.show_basic_info(context.df)
        PlotHelper.plot_target_distribution(context.df, self.output_dir)
        PlotHelper.plot_correlation_heatmap(context.df, self.output_dir)

        print(f"\nGráficos salvos em: {self.output_dir}")

        return context

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path)


    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
        return df