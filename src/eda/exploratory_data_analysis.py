import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["id", "Unnamed: 32"])
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    return df


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


def plot_target_distribution(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(8, 5))
    sns.countplot(x="diagnosis", data=df)
    plt.title("Distribuição do diagnóstico")
    plt.xlabel("Diagnóstico (0 = Benigno, 1 = Maligno)")
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "target_distribution.png"))
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(18, 12))
    sns.heatmap(df.corr(), cmap="coolwarm")
    plt.title("Mapa de Correlação")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()


def run_eda(data_path: str = "../resources/data/breast_cancer.csv",
            output_dir: str = "../resources/outputs/graphs") -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(data_path)
    df = clean_data(df)

    show_basic_info(df)
    plot_target_distribution(df, output_dir)
    plot_correlation_heatmap(df, output_dir)

    print(f"\nGráficos salvos em: {output_dir}")

    return df


if __name__ == "__main__":
    run_eda()
