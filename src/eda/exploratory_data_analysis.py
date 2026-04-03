from pathlib import Path
import pandas as pd
from src.common.print_helper import PrintHelper
from src.common.plot_helper import PlotHelper


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "resources" / "data" / "breast_cancer.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "resources" / "outputs" / "graphs" / "eda"


class ExploratoryDataAnalysis:
    def __init__(
        self,
        data_path: Path = DEFAULT_DATA_PATH,
        output_dir: Path = DEFAULT_OUTPUT_DIR
    ) -> None:
        self.data_path = data_path
        self.output_dir = output_dir


    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path)


    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
        return df


    def run(self) -> pd.DataFrame:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        df = self.load_data()
        df = self.clean_data(df)

        PrintHelper.show_basic_info(df)
        PlotHelper.plot_target_distribution(df, self.output_dir)
        PlotHelper.plot_correlation_heatmap(df, self.output_dir)

        print(f"\nGráficos salvos em: {self.output_dir}")

        return df


def run_eda() -> pd.DataFrame:
    eda = ExploratoryDataAnalysis()
    return eda.run()


if __name__ == "__main__":
    run_eda()