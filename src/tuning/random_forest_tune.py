from typing import Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from src.common.print_helper import PrintHelper
from src.pipeline.pipeline_context import PipelineContext
from src.pipeline.step import Step


class RandomForestTune(Step):
    def __init__(
        self,
        cv: int = 5,
        scoring: str = "recall",
        n_jobs: int = -1,
        random_state: int = 42,
        param_grid: dict[str, list[Any]] | None = None
    ) -> None:
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.param_grid = param_grid or {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }

    def execute(self, context: PipelineContext) -> PipelineContext:
        grid_search = self.create_grid_search()
        grid_search.fit(
            context.data["x_train_scaled"],
            context.data["y_train"]
        )

        context.tuning_results["random_forest"] = {
            "model_name": "random_forest",
            "best_model": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_
        }

        PrintHelper.show_best_result(
            model_name=context.tuning_results["random_forest"]["model_name"],
            best_params=context.tuning_results["random_forest"]["best_params"],
            best_score=context.tuning_results["random_forest"]["best_score"]
        )

        return context

    def create_grid_search(self) -> GridSearchCV:
        return GridSearchCV(
            estimator=RandomForestClassifier(
                random_state=self.random_state
            ),
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs
        )