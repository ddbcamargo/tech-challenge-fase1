from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score


def train_logistic_regression(x_train: Any, y_train: Any) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train, y_train)

    return model


def predict_logistic_regression(model: LogisticRegression, x_test: Any) -> Any:
    return model.predict(x_test)


def evaluate_logistic_regression(model: LogisticRegression, x_test: Any, y_test: Any) -> dict[str, Any]:
    y_pred = predict_logistic_regression(model, x_test)

    return {
        "model_name": "Logistic Regression",
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "predictions": y_pred
    }