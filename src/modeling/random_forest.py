from typing import Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score


def train_random_forest(
    x_train: Any,
    y_train: Any,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(x_train, y_train)

    return model


def predict_random_forest(model: RandomForestClassifier, x_test: Any) -> Any:
    return model.predict(x_test)


def evaluate_random_forest(model: RandomForestClassifier, x_test: Any, y_test: Any) -> dict[str, Any]:
    y_pred = predict_random_forest(model, x_test)

    return {
        "model_name": "Random Forest",
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "predictions": y_pred
    }