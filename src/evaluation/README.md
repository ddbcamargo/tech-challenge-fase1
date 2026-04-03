# 🤖 Modelagem de Machine Learning

## 🎯 Objetivo

A etapa de modelagem tem como objetivo treinar e avaliar modelos de Machine Learning capazes de classificar tumores como benignos ou malignos, com base em dados clínicos.

---

## 📊 Problema

Classificação binária:

- 0 → Benigno  
- 1 → Maligno  

O objetivo é prever corretamente casos malignos, priorizando a redução de falsos negativos.

---

## 🔄 Pipeline adotado

A modelagem foi realizada considerando um pipeline estruturado:

EDA → Preprocessing → Tuning → Modeling → Evaluation

Nesta etapa, os modelos já são utilizados com **hiperparâmetros otimizados previamente na etapa de tuning**.

---

## ⚙️ Modelos utilizados

Foram utilizados três modelos de classificação:

### 1. K-Nearest Neighbors (KNN)

- Algoritmo baseado em distância
- Sensível à escala dos dados
- Utilizado com dados normalizados (StandardScaler)
- Hiperparâmetros ajustados via GridSearchCV

---

### 2. Regressão Logística

- Modelo linear para classificação
- Alta interpretabilidade
- Sensível à escala → dados normalizados
- Hiperparâmetros ajustados via GridSearchCV

---

### 3. Random Forest

- Modelo baseado em árvores de decisão
- Robusto a variáveis correlacionadas
- Não necessita de normalização
- Hiperparâmetros ajustados via GridSearchCV

---

## 🔧 Pré-processamento aplicado

Antes do treinamento:

- Separação entre treino e teste (80/20)
- Aplicação de `StandardScaler` para:
  - KNN
  - Regressão Logística

---

## 🔧 Otimização de hiperparâmetros

A escolha dos melhores parâmetros foi realizada na etapa de **tuning**, utilizando:

- **GridSearchCV**
- **Validação cruzada (cv=5)**
- Métrica de otimização: **Recall**

👉 O recall foi priorizado devido ao contexto médico, onde é mais crítico evitar falsos negativos.

---

## 📈 Métricas utilizadas

Para avaliar os modelos, foram utilizadas:

- **Accuracy** → taxa geral de acertos  
- **Recall** → capacidade de identificar casos malignos  
- **F1-score** → equilíbrio entre precisão e recall  

👉 Em contexto médico, o **recall é especialmente importante**.

---

## 📊 Resultados obtidos

### 🔹 KNN (com tuning)

- Accuracy: **0.9561**
- Recall: **0.9048**
- F1-score: **0.9383**

---

### 🔹 Regressão Logística (com tuning)

- Accuracy: **0.9649**
- Recall: **0.9286**
- F1-score: **0.9512**

---

### 🔹 Random Forest (com tuning)

- Accuracy: **0.9649**
- Recall: **0.9048**
- F1-score: **0.9500**

---

## 🧠 Análise dos resultados

- Todos os modelos apresentaram alto desempenho (acima de 95% de accuracy)
- A utilização de tuning permitiu encontrar configurações mais otimizadas
- A **Regressão Logística apresentou o melhor recall**, sendo o fator decisivo

👉 Interpretação:

A Regressão Logística foi o modelo mais adequado, pois reduz a probabilidade de não identificar casos malignos.

---

## 🏆 Modelo escolhido

**Regressão Logística**

Motivos:

- Melhor recall
- Melhor F1-score
- Alta accuracy
- Fácil interpretação
- Estabilidade após tuning

---

## ⚠️ Limitações

- Dataset relativamente pequeno
- Possível correlação entre variáveis
- Modelo não substitui diagnóstico médico

---

## 🧩 Possíveis melhorias

- Uso de SHAP para explicabilidade
- Feature engineering
- Teste com outros algoritmos (XGBoost, SVM)
- Otimização avançada (Random Search, Bayesian Optimization)

---

## 📌 Conclusão

A etapa de modelagem demonstrou que algoritmos de Machine Learning, quando combinados com técnicas de otimização de hiperparâmetros, são capazes de identificar padrões relevantes em dados clínicos.

A Regressão Logística se destacou como a melhor escolha, principalmente pela sua capacidade de detectar corretamente casos malignos com alta confiabilidade.

---

## ▶️ Execução

```bash
python src/main.py