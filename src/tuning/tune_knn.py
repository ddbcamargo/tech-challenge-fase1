from typing import Any, Dict, Tuple
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def tune_knn(x_train: Any, y_train: Any) -> Tuple[KNeighborsClassifier, Dict[str, Any], float]:
    param_grid = {
        "n_neighbors": list(range(1, 21)),
        "weights": ["uniform", "distance"],
        "metric": ['cosine', 'euclidean', 'manhattan']
    }

    grid_search = GridSearchCV(
        estimator=KNeighborsClassifier(),
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