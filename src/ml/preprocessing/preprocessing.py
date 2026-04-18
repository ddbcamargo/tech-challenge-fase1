from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.ml.pipeline.pipeline_context import PipelineContext
from src.ml.pipeline.step import Step
from src.shared.utils.print_helper import PrintHelper


class Preprocessing(Step):
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state

    def execute(self, context: PipelineContext) -> PipelineContext:
        x, y = self.split_features_target(context.df)
        x_train, x_test, y_train, y_test = self.split_train_test(x, y)
        x_train_scaled, x_test_scaled, scaler = self.scale_data(x_train, x_test)

        context.data = {
            "x": x,
            "y": y,
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
            "x_train_scaled": x_train_scaled,
            "x_test_scaled": x_test_scaled,
            "scaler": scaler,
            "feature_names": list(x.columns),
        }

        PrintHelper.show_shapes(context.data)

        return context

    def split_features_target(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        x = df.drop(columns=["diagnosis"])
        y = df["diagnosis"]
        return x, y

    def split_train_test(
        self,
        x: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        return x_train, x_test, y_train, y_test

    def scale_data(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
    ) -> tuple[Any, Any, StandardScaler]:
        scaler = StandardScaler()

        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        return x_train_scaled, x_test_scaled, scaler
