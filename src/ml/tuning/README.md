# 🔧 Tuning de Hiperparâmetros

## 🎯 Objetivo

A etapa de tuning tem como objetivo encontrar, de forma sistemática, a melhor combinação de hiperparâmetros para cada modelo treinado no projeto, garantindo uma comparação justa entre eles e maximizando o desempenho no problema de classificação de tumores.

---

## 📊 Contexto

Modelos de Machine Learning raramente apresentam bom desempenho com seus parâmetros padrão. Pequenos ajustes em hiperparâmetros (como profundidade de árvores, número de vizinhos ou coeficiente de regularização) podem mudar de forma significativa o resultado final — principalmente em um contexto médico, onde o recall é prioridade.

---

## ⚙️ Técnica utilizada

Foi utilizado o **GridSearchCV** do scikit-learn, que:

- Testa exaustivamente todas as combinações do grid de parâmetros
- Aplica validação cruzada para cada combinação
- Retorna o conjunto que maximiza a métrica escolhida

### 🔹 Configuração adotada

- `cv = 5` → validação cruzada com 5 folds
- `scoring = "recall"` → métrica de otimização
- `n_jobs = -1` → uso de todos os núcleos da CPU

👉 O recall foi escolhido como métrica de otimização por estar diretamente ligado à capacidade do modelo de identificar casos malignos — a prioridade clínica do projeto.

---

## 🔧 Grids de parâmetros

### 1. KNN

```python
{
  "n_neighbors": [1..20],
  "weights": ["uniform", "distance"],
  "metric":  ["cosine", "euclidean", "manhattan"]
}
```

- Explora diferentes tamanhos de vizinhança
- Compara pesagem uniforme versus ponderada pela distância
- Testa três métricas distintas de similaridade

---

### 2. Regressão Logística

```python
{
  "solver": ["lbfgs"],
  "C": [0.01, 0.1, 1, 10, 100]
}
```

- Varia o inverso do coeficiente de regularização (`C`)
- Valores menores → regularização mais forte
- Usa `lbfgs` como solver, adequado para problemas pequenos/médios

---

### 3. Random Forest

```python
{
  "n_estimators": [100, 200, 300],
  "max_depth":    [None, 5, 10, 15],
  "min_samples_split": [2, 5, 10],
  "min_samples_leaf":  [1, 2, 4]
}
```

- Testa diferentes tamanhos de floresta
- Explora profundidades distintas para controlar overfitting
- Ajusta o número mínimo de amostras em splits e folhas

---

## 🔧 Dados utilizados em cada modelo

- **KNN** e **Regressão Logística** → features normalizadas (`x_train_scaled`), por serem sensíveis à escala
- **Random Forest** → features originais (`x_train`), já que modelos baseados em árvore são invariantes a escala

---

## 📦 Saída da etapa

Ao final do tuning, cada modelo contribui para o `PipelineContext` com uma entrada em `context.tuning_results`:

```python
context.tuning_results[model_key] = {
    "model_name":  model_key,
    "best_model":  <estimator ajustado>,
    "best_params": {...},
    "best_score":  <recall médio da cross-validation>,
}
```

Esse estado é consumido pelas etapas seguintes:

- **Modeling** → usa `best_model` para avaliação no conjunto de teste
- **Explainability** → analisa a importância das features do melhor Random Forest
- **Inference** → usa o melhor modelo geral para persistência e inferência via API

---

## 📌 Conclusão

A etapa de tuning é fundamental para uma comparação justa entre os modelos. Sem ela, modelos com parâmetros padrão podem performar aquém do seu potencial real — levando a conclusões enganosas sobre qual algoritmo é mais adequado para o problema.

👉 No projeto atual, o tuning foi decisivo para que o Random Forest superasse a Regressão Logística e se tornasse o modelo vencedor.

---

## ▶️ Execução

```bash
python -m src.main
```
