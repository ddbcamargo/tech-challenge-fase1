from src.eda.exploratory_data_analysis import ExploratoryDataAnalysis
from src.evaluation.evaluation import Evaluation
from src.explainability.shap_analysis import ShapAnalysis
from src.modeling.knn_modeling import KNNModeling
from src.modeling.logistic_regression_modeling import LogisticRegressionModeling
from src.modeling.random_forest_modeling import RandomForestModeling
from src.pipeline.pipeline_context import PipelineContext
from src.pipeline.step import Step
from src.preprocessing.preprocessing import Preprocessing
from src.tuning.knn_tune import KNNTune
from src.tuning.logistic_regression_tune import LogisticRegressionTune
from src.tuning.random_forest_tune import RandomForestTune


class MachineLearningPipeline:
    def __init__(self) -> None:
        self.context = PipelineContext()
        self.steps: list[Step] = [
            ExploratoryDataAnalysis(),
            Preprocessing(),

            KNNTune(),
            KNNModeling(),

            LogisticRegressionTune(),
            LogisticRegressionModeling(),

            RandomForestTune(),
            RandomForestModeling(),

            Evaluation(),
            ShapAnalysis()
        ]

    def run(self) -> PipelineContext:
        for step in self.steps:
            self.context = step.execute(self.context)

        return self.context