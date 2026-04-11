from typing import Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from src.common.print_helper import PrintHelper
from src.pipeline.pipeline_context import PipelineContext
from src.pipeline.step import Step


class RandomForestModeling(Step):
    def execute(self, context: PipelineContext) -> PipelineContext:
        random_forest_tune = context.tuning_results["random_forest"]
        data = context.data

        random_forest_modeling_metrics = self.evaluate(
            random_forest_tune["best_model"],
            data["x_test"],
            data["y_test"]
        )
        context.modeling_results.append(random_forest_modeling_metrics)

        return context

    def predict(self, model: RandomForestClassifier, x_test: Any) -> Any:
        return model.predict(x_test)

    def evaluate(self, model: RandomForestClassifier, x_test: Any, y_test: Any) -> dict[str, Any]:
        y_pred = self.predict(model, x_test)

        result = {
            "model_name": "Random Forest",
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "predictions": y_pred
        }

        PrintHelper.show_metrics(result)

        return result