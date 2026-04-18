from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from src.ml.pipeline.pipeline_context import PipelineContext
from src.ml.pipeline.step import Step
from src.shared.utils.print_helper import PrintHelper


class LogisticRegressionTune(Step):
    def __init__(
        self,
        cv: int = 5,
        scoring: str = "recall",
        n_jobs: int = -1,
        random_state: int = 42,
        max_iter: int = 2000,
        param_grid: dict[str, list[Any]] | None = None,
    ) -> None:
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_iter = max_iter

        self.param_grid = param_grid or {
            "solver": ["lbfgs"],
            "C": [0.01, 0.1, 1, 10, 100],
        }

    def execute(self, context: PipelineContext) -> PipelineContext:
        grid_search = self.create_grid_search()
        grid_search.fit(
            context.data["x_train_scaled"],
            context.data["y_train"],
        )

        context.tuning_results["logistic_regression"] = {
            "model_name": "logistic_regression",
            "best_model": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
        }

        PrintHelper.show_best_result(
            model_name=context.tuning_results["logistic_regression"]["model_name"],
            best_params=context.tuning_results["logistic_regression"]["best_params"],
            best_score=context.tuning_results["logistic_regression"]["best_score"],
        )

        return context

    def create_grid_search(self) -> GridSearchCV:
        return GridSearchCV(
            estimator=LogisticRegression(
                max_iter=self.max_iter,
                random_state=self.random_state,
            ),
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )
