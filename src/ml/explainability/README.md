# 🔍 Explainability (Explicabilidade do Modelo)

## 🎯 Objetivo

A etapa de explainability tem como objetivo interpretar o comportamento do modelo de Machine Learning, tornando suas decisões mais transparentes e compreensíveis.

Isso é especialmente importante em contextos médicos, onde a confiabilidade do modelo é essencial.

---

## 🧠 Abordagens utilizadas

Foram utilizadas duas técnicas principais:

### 1. Feature Importance (Random Forest)

Mede a importância de cada variável com base na redução de impureza nas árvores.

---

### 2. SHAP (SHapley Additive Explanations)

Técnica baseada em teoria dos jogos que explica:

- impacto de cada variável na previsão
- direção da influência (positiva ou negativa)
- comportamento global do modelo

---

## 📊 Feature Importance

Principais variáveis identificadas:

- `area_worst`
- `concave points_worst`
- `perimeter_worst`
- `radius_worst`

👉 Essas variáveis indicam que o modelo utiliza principalmente:

- tamanho do tumor
- irregularidade da forma

---

## 📈 SHAP - Análise Global

### 📊 Summary Bar

Mostra a importância média das variáveis:

- confirma os resultados do Feature Importance
- reforça que `area_worst` é a variável mais relevante

---

### 📊 Beeswarm Plot

Mostra:

- impacto de cada variável nas previsões
- distribuição dos valores
- direção da influência

#### 🔎 Interpretação:

- pontos vermelhos (valores altos) geralmente empurram a previsão para **maligno**
- pontos azuis (valores baixos) tendem para **benigno**

---

## 🧠 Insights do Modelo

O modelo aprendeu padrões coerentes com a prática clínica:

### ✔ Tumores malignos tendem a:

- ter maior tamanho (`area`, `radius`, `perimeter`)
- possuir bordas irregulares (`concavity`, `concave points`)

👉 Isso valida que o modelo está aprendendo relações reais do problema.

---

## 🔍 Confiabilidade do Modelo

A combinação de:

- alta performance (accuracy, recall, f1-score)
- interpretabilidade (SHAP + Feature Importance)

aumenta significativamente a confiança no modelo.

---

## 📌 Conclusão

A análise de explainability mostrou que o modelo toma decisões baseadas em características relevantes e coerentes com o domínio médico.

Isso torna o modelo não apenas eficiente, mas também confiável e interpretável.

---

## ▶️ Execução

```bash
python -m src.main