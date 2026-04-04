from typing import Any
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from src.common.print_helper import PrintHelper
from src.pipeline.pipeline_context import PipelineContext
from src.pipeline.step import Step


class KNNModeling(Step):
    def execute(self, context: PipelineContext) -> PipelineContext:
        knn_tune = context.tuning_results["knn"]
        data = context.data

        knn_modeling_metrics = self.evaluate(
            knn_tune["best_model"],
            data["x_test_scaled"],
            data["y_test"]
        )
        context.modeling_results.append(knn_modeling_metrics)

        return context

    def predict(self, model: KNeighborsClassifier, x_test: Any) -> Any:
        return model.predict(x_test)

    def evaluate(self, model: KNeighborsClassifier, x_test: Any, y_test: Any) -> dict[str, Any]:
        y_pred = self.predict(model, x_test)

        result = {
            "model_name": "KNN",
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "predictions": y_pred
        }

        PrintHelper.show_metrics(result)

        return result