from pathlib import Path


BASE_DIR: Path = Path(__file__).resolve().parents[3]

RESOURCES_DIR: Path = BASE_DIR / "resources"
DATA_DIR: Path = RESOURCES_DIR / "data"
OUTPUTS_DIR: Path = RESOURCES_DIR / "outputs"
GRAPHS_DIR: Path = OUTPUTS_DIR / "graphs"

EDA_GRAPHS_DIR: Path = GRAPHS_DIR / "eda"
EVALUATION_GRAPHS_DIR: Path = GRAPHS_DIR / "evaluation"
EXPLAINABILITY_GRAPHS_DIR: Path = GRAPHS_DIR / "explainability"

MODELS_DIR: Path = BASE_DIR / "models"
ML_MODELS_DIR: Path = MODELS_DIR / "ml"

DEFAULT_DATASET_PATH: Path = DATA_DIR / "breast_cancer.csv"
