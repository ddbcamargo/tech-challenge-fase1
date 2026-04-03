from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    return x, y


def split_train_test(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return x_train, x_test, y_train, y_test


def scale_data(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame
) -> tuple[Any, Any, StandardScaler]:
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, scaler


def run_preprocessing(df: pd.DataFrame) -> dict[str, Any]:
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = split_train_test(x, y)
    x_train_scaled, x_test_scaled, scaler = scale_data(x_train, x_test)

    print("=== PREPROCESSING ===")
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return {
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