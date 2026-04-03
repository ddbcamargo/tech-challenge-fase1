from typing import Any
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier


def train_knn(x_train: Any, y_train: Any, n_neighbors: int = 5) -> KNeighborsClassifier:
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)

    return model


def predict_knn(model: KNeighborsClassifier, x_test: Any) -> Any:
    return model.predict(x_test)


def evaluate_knn(model: KNeighborsClassifier, x_test: Any, y_test: Any) -> dict[str, Any]:
    y_pred = predict_knn(model, x_test)

    return {
        "model_name": "KNN",
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "predictions": y_pred
    }