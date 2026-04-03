from src.pipeline.pipeline_context import PipelineContext

from src.eda.exploratory_data_analysis import run_eda

from src.preprocessing.preprocessing import run_preprocessing

from src.tuning.knn_tune import run_knn_tune
from src.modeling.knn_modeling import run_knn_modeling

from src.tuning.logistic_regression_tune import run_logistic_regression_tune
from src.modeling.logistic_regression_modeling import run_logistic_regression_modeling

from src.tuning.random_forest_tune import run_random_forest_tune
from src.modeling.random_forest_modeling import run_random_forest_modeling

from src.evaluation.evaluation import run_evaluation

from src.explainability.feature_importance import run_feature_importance
from src.explainability.shap_analysis import run_shap_analysis


class MachineLearningPipeline:
    def __init__(self) -> None:
        self.context = PipelineContext()

    #EDA
    def run_eda(self):
        self.context.df = run_eda()
        return self

    #PREPROCESSING
    def run_preprocessing(self):
        self.context.data = run_preprocessing(self.context.df)
        return self

    #TUNING AND MODELING
    def run_knn(self):
        data = self.context.data

        knn_tune = run_knn_tune(data["x_train_scaled"], data["y_train"])
        knn_metrics = run_knn_modeling(knn_tune["best_model"], data["x_test_scaled"], data["y_test"])

        self.context.tuning_results["knn"] = knn_tune
        self.context.modeling_results.append(knn_metrics)

        return self

    def run_logistic_regression(self):
        data = self.context.data

        logistic_regression_tune = run_logistic_regression_tune(data["x_train_scaled"], data["y_train"])
        logistic_regression_metrics = run_logistic_regression_modeling(logistic_regression_tune["best_model"], data["x_test_scaled"], data["y_test"])

        self.context.tuning_results["logistic_regression"] = logistic_regression_tune
        self.context.modeling_results.append(logistic_regression_metrics)

        return self

    def run_random_forest(self):
        data = self.context.data

        random_forest_tune = run_random_forest_tune(data["x_train"], data["y_train"])
        random_forest_metrics = run_random_forest_modeling(random_forest_tune["best_model"], data["x_test"], data["y_test"])

        self.context.tuning_results["random_forest"] = random_forest_tune
        self.context.modeling_results.append(random_forest_metrics)

        return self

    #EVALUATION
    def run_evaluation(self):
        self.context.evaluation_df = run_evaluation(
            self.context.modeling_results,
            self.context.data["y_test"]
        )

        return self

    #EXPLAINABILITY
    def run_explainability(self):
        model = self.context.tuning_results["random_forest"]["best_model"]
        x_train = self.context.data["x_train"]

        run_feature_importance(
            model=model,
            feature_names=list(x_train.columns),
            top_n=10
        )

        run_shap_analysis(
            model=model,
            x_data=x_train,
            feature_names=list(x_train.columns)
        )

        return self

def run_machine_learning_pipeline():
    machine_learning_pipeline = MachineLearningPipeline()

    return (
        machine_learning_pipeline
            .run_eda()
            .run_preprocessing()
            .run_knn()
            .run_logistic_regression()
            .run_random_forest()
            .run_evaluation()
            .run_explainability()
    )