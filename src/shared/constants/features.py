"""Constantes referentes ao dataset de câncer de mama."""

TARGET_COLUMN: str = "diagnosis"

CLASS_LABELS: dict[int, str] = {
    0: "benigno",
    1: "maligno",
}

# Ordem exata das features esperadas pelo modelo treinado.
# Mantida explicitamente para garantir consistência em treino e inferência
# (API + pipelines).
FEATURE_COLUMNS: list[str] = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

# Mapeamento entre os nomes "amigáveis" aceitos pela API (snake_case)
# e os nomes reais das colunas do dataset.
API_TO_DATASET_FEATURE: dict[str, str] = {
    name.replace(" ", "_"): name for name in FEATURE_COLUMNS
}
