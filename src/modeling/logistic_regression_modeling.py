from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from src.common.print_helper import PrintHelper
from src.pipeline.pipeline_context import PipelineContext
from src.pipeline.step import Step


class LogisticRegressionModeling(Step):
    def execute(self, context: PipelineContext) -> PipelineContext:
        logistic_regression_tune = context.tuning_results["logistic_regression"]
        data = context.data

        logistic_regression_modeling_metrics = self.evaluate(
            logistic_regression_tune["best_model"],
            data["x_test_scaled"],
            data["y_test"]
        )
        context.modeling_results.append(logistic_regression_modeling_metrics)

        return context

    def predict(self, model: LogisticRegression, x_test: Any) -> Any:
        return model.predict(x_test)

    def evaluate(self, model: LogisticRegression, x_test: Any, y_test: Any) -> dict[str, Any]:
        y_pred = self.predict(model, x_test)

        result = {
            "model_name": "Logistic Regression",
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "predictions": y_pred
        }

        PrintHelper.show_metrics(result)

        return result