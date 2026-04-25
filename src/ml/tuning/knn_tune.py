from typing import Any

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from src.ml.pipeline.pipeline_context import PipelineContext
from src.ml.pipeline.step import Step
from src.shared.utils.print_helper import PrintHelper


class KNNTune(Step):
    def __init__(
        self,
        cv: int = 5,
        scoring: str = "recall",
        n_jobs: int = -1,
        param_grid: dict[str, list[Any]] | None = None,
    ) -> None:
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.param_grid = param_grid or {
            "n_neighbors": list(range(1, 21)),
            "weights": ["uniform", "distance"],
            "metric": ["cosine", "euclidean", "manhattan"],
        }

    def execute(self, context: PipelineContext) -> PipelineContext:
        grid_search = self.create_grid_search()
        grid_search.fit(
            context.data["x_train_scaled"],
            context.data["y_train"],
        )

        context.tuning_results["knn"] = {
            "model_name": "knn",
            "best_model": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
        }

        PrintHelper.show_best_result(
            model_name=context.tuning_results["knn"]["model_name"],
            best_params=context.tuning_results["knn"]["best_params"],
            best_score=context.tuning_results["knn"]["best_score"],
        )

        return context

    def create_grid_search(self) -> GridSearchCV:
        return GridSearchCV(
            estimator=KNeighborsClassifier(),
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )
