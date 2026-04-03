from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.common.print_helper import PrintHelper


class Preprocessing:
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state

    def split_features_target(
        self,
        df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        x = df.drop(columns=["diagnosis"])
        y = df["diagnosis"]
        return x, y

    def split_train_test(
        self,
        x: pd.DataFrame,
        y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        return x_train, x_test, y_train, y_test

    def scale_data(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame
    ) -> tuple[Any, Any, StandardScaler]:
        scaler = StandardScaler()

        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        return x_train_scaled, x_test_scaled, scaler

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        x, y = self.split_features_target(df)
        x_train, x_test, y_train, y_test = self.split_train_test(x, y)
        x_train_scaled, x_test_scaled, scaler = self.scale_data(x_train, x_test)

        result = {
            "x": x,
            "y": y,
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
            "x_train_scaled": x_train_scaled,
            "x_test_scaled": x_test_scaled,
            "scaler": scaler
        }

        PrintHelper.show_shapes(result)

        return result


def run_preprocessing(df: pd.DataFrame) -> dict[str, Any]:
    preprocessing = Preprocessing()
    return preprocessing.run(df)