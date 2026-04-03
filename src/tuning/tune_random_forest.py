from typing import Any, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def tune_random_forest(x_train: Any, y_train: Any) -> Tuple[RandomForestClassifier, Dict[str, Any], float]:
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
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