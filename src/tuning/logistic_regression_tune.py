from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from src.common.print_helper import PrintHelper


class TuneLogisticRegression:
    def __init__(
        self,
        cv: int = 5,
        scoring: str = "recall",
        n_jobs: int = -1,
        random_state: int = 42,
        max_iter: int = 2000,
        param_grid: dict[str, list[Any]] | None = None
    ) -> None:
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_iter = max_iter

        self.param_grid = param_grid or {
            "solver": ["lbfgs"],
            "C": [0.01, 0.1, 1, 10, 100]
        }

    def create_grid_search(self) -> GridSearchCV:
        return GridSearchCV(
            estimator=LogisticRegression(
                max_iter=self.max_iter,
                random_state=self.random_state
            ),
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs
        )

    def run(self, x_train: Any, y_train: Any) -> dict[str, Any]:
        grid_search = self.create_grid_search()
        grid_search.fit(x_train, y_train)

        result = {
            "model_name": "logistic_regression",
            "best_model": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_
        }

        PrintHelper.show_best_result(
            model_name=result["model_name"],
            best_params=result["best_params"],
            best_score=result["best_score"]
        )

        return result


def run_logistic_regression_tune(x_train: Any, y_train: Any) -> dict[str, Any]:
    tuner = TuneLogisticRegression()
    return tuner.run(x_train, y_train)