from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)

from src.ml.pipeline.pipeline_context import PipelineContext
from src.ml.pipeline.step import Step
from src.shared.utils.print_helper import PrintHelper


class RandomForestModeling(Step):
    MODEL_KEY = "random_forest"
    DISPLAY_NAME = "Random Forest"

    def execute(self, context: PipelineContext) -> PipelineContext:
        tune = context.tuning_results[self.MODEL_KEY]
        data = context.data

        metrics = self.evaluate(
            tune["best_model"],
            data["x_test"],
            data["y_test"],
        )
        metrics["model_key"] = self.MODEL_KEY
        # Random Forest trabalha diretamente nos dados não normalizados.
        metrics["requires_scaler"] = False
        context.modeling_results.append(metrics)

        return context

    def predict(self, model: RandomForestClassifier, x_test: Any) -> Any:
        return model.predict(x_test)

    def evaluate(
        self,
        model: RandomForestClassifier,
        x_test: Any,
        y_test: Any,
    ) -> dict[str, Any]:
        y_pred = self.predict(model, x_test)

        result = {
            "model_name": self.DISPLAY_NAME,
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "predictions": y_pred,
        }

        PrintHelper.show_metrics(result)

        return result
