from typing import Any, Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def tune_logistic_regression(x_train: Any, y_train: Any) -> Tuple[LogisticRegression, Dict[str, Any], float]:
    param_grid = {
        "solver": ["lbfgs"],
        "C": [0.01, 0.1, 1, 10, 100]
    }

    grid_search = GridSearchCV(
        estimator=LogisticRegression(max_iter=2000, random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1
    )

    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score