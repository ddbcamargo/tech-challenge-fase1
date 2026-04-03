from typing import Any
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from src.common.print_helper import PrintHelper


class KNNModeling:
    def __init__(self, model: KNeighborsClassifier) -> None:
        self.model = model

    def predict(self, x_test: Any) -> Any:
        return self.model.predict(x_test)

    def evaluate(self, x_test: Any, y_test: Any) -> dict[str, Any]:
        y_pred = self.predict(x_test)

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

    def run(self, x_test: Any, y_test: Any) -> dict[str, Any]:
        return self.evaluate(x_test, y_test)


def run_knn_modeling(model: KNeighborsClassifier, x_test: Any, y_test: Any) -> dict[str, Any]:
    modeling = KNNModeling(model)
    return modeling.run(x_test, y_test)