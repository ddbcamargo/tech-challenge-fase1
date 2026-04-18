from src.ml.eda.exploratory_data_analysis import ExploratoryDataAnalysis
from src.ml.evaluation.evaluation import Evaluation
from src.ml.explainability.shap_analysis import ShapAnalysis
from src.ml.inference.persist_best_model import PersistBestModel
from src.ml.modeling.knn_modeling import KNNModeling
from src.ml.modeling.logistic_regression_modeling import LogisticRegressionModeling
from src.ml.modeling.random_forest_modeling import RandomForestModeling
from src.ml.pipeline.pipeline_context import PipelineContext
from src.ml.pipeline.step import Step
from src.ml.preprocessing.preprocessing import Preprocessing
from src.ml.tuning.knn_tune import KNNTune
from src.ml.tuning.logistic_regression_tune import LogisticRegressionTune
from src.ml.tuning.random_forest_tune import RandomForestTune


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
            ShapAnalysis(),

            # Seleciona o melhor modelo com base nas métricas e persiste
            # artefatos (modelo, scaler, metadados) em models/ml/ para que
            # a API de inferência consiga carregar sem acoplar-se ao treino.
            PersistBestModel(),
        ]

    def run(self) -> PipelineContext:
        for step in self.steps:
            self.context = step.execute(self.context)

        return self.context
