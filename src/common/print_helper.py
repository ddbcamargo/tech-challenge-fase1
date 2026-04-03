from typing import Any

import pandas as pd


class PrintHelper:
    @staticmethod
    def show_basic_info(df: pd.DataFrame) -> None:
        print("=== SHAPE ===")
        print(df.shape)

        print("\n=== INFO ===")
        df.info()

        print("\n=== DESCRIBE ===")
        print(df.describe())

        print("\n=== MISSING VALUES ===")
        print(df.isnull().sum())

        print("\n=== TARGET DISTRIBUTION ===")
        print(df["diagnosis"].value_counts())

    @staticmethod
    def show_shapes(data: dict) -> None:
        print("=== PREPROCESSING ===")
        print(f"x shape: {data['x'].shape}")
        print(f"y shape: {data['y'].shape}")
        print(f"x_train shape: {data['x_train'].shape}")
        print(f"x_test shape: {data['x_test'].shape}")
        print(f"y_train shape: {data['y_train'].shape}")
        print(f"y_test shape: {data['y_test'].shape}")

    @staticmethod
    def show_best_result(
            model_name: str,
            best_params: dict[str, Any],
            best_score: float
    ) -> None:
        print(f"=== TUNING: {model_name.upper()} ===")
        print(f"Best params: {best_params}")
        print(f"Best recall: {best_score:.4f}")

    @staticmethod
    def show_metrics(result: dict[str, Any]) -> None:
        print(f"\n=== {result['model_name']} ===")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print(f"F1-Score: {result['f1_score']:.4f}")
        print("\nClassification Report:")
        print(result["classification_report"])

    @staticmethod
    def print_metrics_summary(metrics_df: pd.DataFrame) -> None:
        print("\n=== MODEL COMPARISON ===")
        print(
            metrics_df
                .sort_values(by=["recall", "f1_score", "accuracy"], ascending=False)
                .to_string(index=False)
        )

    @staticmethod
    def print_feature_importance(
            importance_df: pd.DataFrame,
            top_n: int = 10
    ) -> None:
        print(f"\n=== TOP {top_n} FEATURE IMPORTANCE - RANDOM FOREST ===")
        print(importance_df.head(top_n).to_string(index=False))