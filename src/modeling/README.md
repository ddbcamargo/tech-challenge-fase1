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

EDA → Preprocessing → Tuning → Modeling → Evaluation

Nesta etapa, os modelos são avaliados utilizando **hiperparâmetros previamente otimizados na etapa de tuning**.

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
- Boa interpretabilidade
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

A escolha dos melhores parâmetros foi realizada utilizando:

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

👉 Em contexto médico, o **recall é especialmente importante**, pois reduz o risco de falsos negativos.

---

## 📊 Resultados obtidos (com tuning)

### 🔹 Random Forest

- Accuracy: **0.9737**
- Recall: **0.9286**
- F1-score: **0.9630**

---

### 🔹 Regressão Logística

- Accuracy: **0.9474**
- Recall: **0.8810**
- F1-score: **0.9250**

---

### 🔹 KNN

- Accuracy: **0.9298**
- Recall: **0.8571**
- F1-score: **0.9000**

---

## 🧠 Análise dos resultados

- A aplicação de tuning impactou diretamente o desempenho dos modelos
- O **Random Forest apresentou o melhor desempenho geral**
- O modelo obteve o maior recall, sendo o mais adequado para o problema

👉 Interpretação:

Após otimização, o Random Forest conseguiu capturar melhor relações não lineares presentes nos dados, aumentando a capacidade de identificar corretamente casos malignos.

---

## 🔄 Comparação antes vs depois do tuning

- Sem tuning → Regressão Logística apresentou melhor desempenho  
- Com tuning → Random Forest se tornou o melhor modelo  

👉 Isso evidencia a importância da otimização de hiperparâmetros para comparação justa entre modelos.

---

## 🏆 Modelo escolhido

**Random Forest**

Motivos:

- Melhor recall (métrica prioritária)
- Melhor F1-score
- Maior accuracy
- Capacidade de capturar relações não lineares
- Melhor desempenho após tuning

---

## ⚠️ Limitações

- Dataset relativamente pequeno
- Possível correlação entre variáveis
- Modelo não substitui diagnóstico médico

---

## 📌 Conclusão

A etapa de modelagem demonstrou que algoritmos de Machine Learning, quando combinados com técnicas de otimização de hiperparâmetros, são capazes de identificar padrões relevantes em dados clínicos.

O modelo Random Forest se destacou como a melhor escolha, principalmente pela sua capacidade de detectar corretamente casos malignos com maior confiabilidade.

---

## ▶️ Execução

```bash
python -m src.main