# 🧠 Breast Cancer Classification - Machine Learning

## ▶️ Como executar o projeto
### 🔹 1. Clonar o repositório
````
git clone git@github.com:ddbcamargo/tech-challenge-fase1.git
````

### 🔹 2. Criar ambiente virtual
````bash
python -m venv .venv
````

#### Ativar:
- Windows
````bash
.venv\Scripts\activate
````
- Linux/Mac
````bash
source .venv/bin/activate
````

### 🔹 3. Instalar dependências
````bash
pip install -r requirements.txt
````

### 🔹 4. Executar o projeto

O entrypoint aceita três formas de execução:

| Comando                              | Comportamento                                                   |
| ------------------------------------ | --------------------------------------------------------------- |
| `python -m src.main`                 | Executa **ML + API em sequência** (treina e já sobe a API)      |
| `python -m src.main --mode ml`       | Executa **apenas o pipeline de Machine Learning** (treino)      |
| `python -m src.main --mode api`      | Sobe **apenas a API Flask** (usa os artefatos já treinados)     |

#### Exemplos

````bash
# Treina o modelo e sobe a API logo em seguida (default)
python -m src.main

# Apenas treina (gera best_model.joblib, scaler.joblib, best_model_info.json)
python -m src.main --mode ml

# Apenas sobe a API Flask em http://localhost:5000
python -m src.main --mode api
````

> **Observação:** o modo `--mode api` exige que o pipeline de ML já tenha sido
> executado pelo menos uma vez (para que os artefatos em `models/ml/` existam).
> Em uma máquina nova, prefira rodar `python -m src.main` (sem flag) na
> primeira execução.

### 🔹 5. API Flask + Swagger

A API HTTP expõe o melhor modelo treinado para inferência:

| Método | Rota            | Descrição                                    |
| ------ | --------------- | -------------------------------------------- |
| GET    | `/`             | Redireciona para o Swagger UI                |
| GET    | `/apidocs/`     | **Swagger UI** (documentação interativa)     |
| GET    | `/apispec.json` | Spec OpenAPI 2.0 em JSON                     |
| GET    | `/health`       | Health check + info do modelo carregado      |
| POST   | `/predict`      | Predição binária (benigno/maligno) + prob.   |

#### Exemplo de uso

Request body em `POST /predict`:

```json
{
  "radius_mean": 14.2,
  "texture_mean": 20.1,
  "perimeter_mean": 92.0,
  "area_mean": 654.0,
  "smoothness_mean": 0.09,
  "compactness_mean": 0.1,
  "concavity_mean": 0.08,
  "concave_points_mean": 0.04,
  "symmetry_mean": 0.18,
  "fractal_dimension_mean": 0.06,
  "radius_se": 0.5,
  "texture_se": 1.2,
  "perimeter_se": 3.5,
  "area_se": 40.0,
  "smoothness_se": 0.007,
  "compactness_se": 0.02,
  "concavity_se": 0.03,
  "concave_points_se": 0.01,
  "symmetry_se": 0.02,
  "fractal_dimension_se": 0.003,
  "radius_worst": 16.5,
  "texture_worst": 25.0,
  "perimeter_worst": 110.0,
  "area_worst": 880.0,
  "smoothness_worst": 0.13,
  "compactness_worst": 0.22,
  "concavity_worst": 0.25,
  "concave_points_worst": 0.12,
  "symmetry_worst": 0.28,
  "fractal_dimension_worst": 0.09
}
```

Response:

```json
{
  "prediction": 1,
  "label": "maligno",
  "probability": 0.93,
  "model": "random_forest"
}
```

A forma mais fácil de testar é abrir
[http://localhost:5000/apidocs/](http://localhost:5000/apidocs/) e usar
o botão *Try it out* — o Swagger UI já vem com o payload acima
preenchido.


---

## 📌 Visão Geral

Este projeto implementa um pipeline completo de Machine Learning para classificação de tumores como **benignos** ou **malignos**, utilizando dados clínicos do dataset Breast Cancer.

O objetivo é construir um modelo confiável, interpretável e alinhado com o contexto médico, priorizando a identificação correta de casos malignos.

---

## 🎯 Objetivo do Projeto

Desenvolver um modelo de classificação capaz de:

- Identificar tumores malignos com alta precisão
- Minimizar falsos negativos (casos perigosos)
- Fornecer explicabilidade das decisões do modelo

---

## 🧱 Estrutura do Projeto

```bash
src/
├── ml/
│   ├── eda/
│   ├── preprocessing/
│   ├── modeling/
│   ├── tuning/
│   ├── evaluation/
│   ├── explainability/
│   ├── pipeline/
│   └── inference/
├── api/
├── shared/
└── main.py
```

#### 🔹 EDA (Exploratory Data Analysis)
- Análise inicial dos dados
- Distribuição da variável target
- Identificação de padrões e correlações

👉 Mais detalhes em [README.md](src/ml/eda/README.md)

#### 🔹 Preprocessing
- Limpeza de dados
- Conversão da variável target (M/B → 1/0)
- Separação entre treino e teste (80/20)
- Normalização com StandardScaler

👉 Mais detalhes em [README.md](src/ml/preprocessing/README.md)

#### 🔹 Modeling
Modelos treinados:
- KNN (K-Nearest Neighbors)
- Regressão Logística
- Random Forest

👉 Mais detalhes em [README.md](src/ml/modeling/README.md)

#### 🔹 Tuning
Ajuste de hiperparâmetros com:
- GridSearchCV
- Validação cruzada (cross-validation)

👉 Mais detalhes em [README.md](src/ml/tuning/README.md)

#### 🔹 Evaluation
Métricas utilizadas:
- Accuracy
- Recall (principal métrica)
- F1-score
- Classification Report

👉 Mais detalhes em [README.md](src/ml/evaluation/README.md)

#### 🔹 Explainability
- Importance (Random Forest)
- SHAP (interpretação avançada do modelo)

👉 Mais detalhes em [README.md](src/ml/explainability/README.md)

#### 🔹 Pipeline
- Orquestração das etapas (EDA → Preprocessing → Tuning → Modeling → Evaluation → Explainability → Inference)
- `Step` (contrato abstrato) + `PipelineContext` (estado compartilhado)
- `MachineLearningPipeline` como executor principal

👉 Mais detalhes em [README.md](src/ml/pipeline/README.md)

#### 🔹 Inference
- Seleção do melhor modelo (recall → f1 → accuracy)
- Persistência de `best_model.joblib`, `scaler.joblib` e `best_model_info.json`
- `MLPredictor`: classe consumida pela API Flask

👉 Mais detalhes em [README.md](src/ml/inference/README.md)

---

## 📊 Problema de Classificação
- 0 → Benigno
- 1 → Maligno

⚠️ O foco do projeto é reduzir falsos negativos, pois eles representam maior risco no contexto médico.

___

## 🏆 Resultado Final
### 📊 Comparação de Modelos (após tuning)
| Modelo              | Accuracy | Recall | F1-score |
| ------------------- | -------- | ------ | -------- |
| Random Forest       | 0.9737   | 0.9286 | 0.9630   |
| Logistic Regression | 0.9474   | 0.8810 | 0.9250   |
| KNN                 | 0.9298   | 0.8571 | 0.9000   |

---

## 🥇 Modelo Escolhido
### Random Forest
#### Motivos:
- Melhor desempenho geral
- Alto recall (reduz falsos negativos)
- Maior robustez
- Melhor desempenho após tuning

---

## 🧠 Explainability (Interpretação do Modelo)
### 🔹 Feature Importance
#### Principais variáveis:
- area_worst
- concave points_worst
- perimeter_worst
- radius_worst

### 🔹 SHAP
#### A análise com SHAP mostrou que:
- Valores altos de certas features aumentam a chance de malignidade
- O modelo baseia suas decisões em padrões coerentes

---

## 🔬 Insights do Modelo
O modelo aprendeu padrões clínicos relevantes:
#### Tumores malignos tendem a:
- ser maiores (area, radius, perimeter)
- ter formas irregulares (concavity, concave points)

👉 Isso aumenta a confiabilidade da solução.

---

## 📁 Outputs Gerados

Os resultados são salvos em:
````
resources/outputs/
````
Incluindo:
- gráficos de EDA
- feature importance
- SHAP plots
- resultados dos modelos

---

## 🧠 Tecnologias Utilizadas
- Python
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- SHAP
- Flask (API de inferência)
- Flasgger (Swagger UI)
- Joblib (persistência de modelo e scaler)

---

## 📚 Documentação

- [EDA](src/ml/eda/README.md)
- [Preprocessing](src/ml/preprocessing/README.md)
- [Modeling](src/ml/modeling/README.md)
- [Tuning](src/ml/tuning/README.md)
- [Evaluation](src/ml/evaluation/README.md)
- [Explainability](src/ml/explainability/README.md)
- [Pipeline](src/ml/pipeline/README.md)
- [Inference](src/ml/inference/README.md)
---

## 📌 Conclusão
O projeto demonstrou que técnicas de Machine Learning podem identificar padrões relevantes em dados clínicos, auxiliando no diagnóstico precoce de câncer de mama.

O uso de técnicas de explicabilidade garante que o modelo seja não apenas performático, mas também confiável.

---

## 👨‍💻 Autor

Diego Diondré Bueno de Camargo (diego.diondre@gmail.com)
