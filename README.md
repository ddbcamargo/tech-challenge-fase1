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

### 🔹 4. Executar o pipeline
````bash
python -m src.main
````

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
├── eda/
├── preprocessing/
├── modeling/
├── tuning/
├── evaluation/
├── explainability/
└── main.py
```

#### 🔹 EDA (Exploratory Data Analysis)
- Análise inicial dos dados
- Distribuição da variável target
- Identificação de padrões e correlações

👉 Mais detalhes em [README.md](src/eda/README.md)

#### 🔹 Preprocessing
- Limpeza de dados
- Conversão da variável target (M/B → 1/0)
- Separação entre treino e teste (80/20)
- Normalização com StandardScaler

👉 Mais detalhes em [README.md](src/preprocessing/README.md)

#### 🔹 Modeling
Modelos treinados:
- KNN (K-Nearest Neighbors)
- Regressão Logística
- Random Forest

👉 Mais detalhes em [README.md](src/modeling/README.md)

#### 🔹 Tuning
Ajuste de hiperparâmetros com:
- GridSearchCV
- Validação cruzada (cross-validation)

👉 Mais detalhes em [README.md](src/tuning/README.md)

#### 🔹 Evaluation
Métricas utilizadas:
- Accuracy
- Recall (principal métrica)
- F1-score
- Classification Report

👉 Mais detalhes em [README.md](src/evaluation/README.md)

#### 🔹 Explainability
- Importance (Random Forest)
- SHAP (interpretação avançada do modelo)

👉 Mais detalhes em [README.md](src/explainability/README.md)

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

---

## 📚 Documentação

- [EDA](src/eda/README.md)
- [Preprocessing](src/preprocessing/README.md)
- [Modeling](src/modeling/README.md)
- [Tuning](src/tuning/README.md)
- [Evaluation](src/evaluation/README.md)
- [Explainability](src/explainability/README.md)
---

## 📌 Conclusão
O projeto demonstrou que técnicas de Machine Learning podem identificar padrões relevantes em dados clínicos, auxiliando no diagnóstico precoce de câncer de mama.

O uso de técnicas de explicabilidade garante que o modelo seja não apenas performático, mas também confiável.

---

## 👨‍💻 Autor

Diego Diondré Bueno de Camargo (diego.diondre@gmail.com)