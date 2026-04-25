# 🚀 Inference

## 🎯 Objetivo

A etapa de inferência tem como objetivo **selecionar o melhor modelo treinado**, **persistir os artefatos necessários em disco** e **expor um ponto único de predição** para ser consumido pela API Flask.

É a ponte entre o mundo do treinamento (pipeline de ML) e o mundo da produção (API HTTP).

---

## 🧱 Componentes

A camada de inferência é composta por três peças que trabalham em sequência:

### 🔹 `best_model_selector.py` — `select_best_model`

Função responsável por escolher, entre os modelos avaliados, aquele que será persistido como "modelo vencedor".

```python
ranked = sorted(
    modeling_results,
    key=lambda r: (r["recall"], r["f1_score"], r["accuracy"]),
    reverse=True,
)
return ranked[0]
```

#### Critério de desempate

A ordenação usa uma tupla `(recall, f1_score, accuracy)` em ordem decrescente:

1. **Recall** → métrica principal (minimizar falsos negativos em contexto médico)
2. **F1-score** → equilíbrio entre precisão e recall
3. **Accuracy** → desempate final

👉 Essa estratégia reflete a prioridade do domínio: **nunca deixar passar um tumor maligno**.

---

### 🔹 `persist_best_model.py` — `PersistBestModel`

Step do pipeline executado **após a avaliação**. Grava em `models/ml/` três artefatos essenciais:

| Arquivo                 | Conteúdo                                                    |
| ----------------------- | ----------------------------------------------------------- |
| `best_model.joblib`     | Estimador sklearn treinado (o vencedor)                     |
| `scaler.joblib`         | `StandardScaler` ajustado no treino                         |
| `best_model_info.json`  | Metadados: nome, métricas, features, flag `requires_scaler` |

Exemplo de `best_model_info.json`:

```json
{
  "model_key": "random_forest",
  "model_name": "Random Forest",
  "requires_scaler": false,
  "metrics": {
    "accuracy": 0.9737,
    "recall": 0.9286,
    "f1_score": 0.9630
  },
  "feature_names": ["radius_mean", "texture_mean", "..."],
  "artifacts": {
    "model": "best_model.joblib",
    "scaler": "scaler.joblib"
  }
}
```

👉 A flag `requires_scaler` informa se o modelo precisa (KNN, Regressão Logística) ou não (Random Forest) da normalização no momento da predição.

---

### 🔹 `predictor.py` — `MLPredictor`

Classe que encapsula o carregamento dos artefatos e expõe um método `predict` pronto para ser consumido pela API.

```python
class MLPredictor:
    def __init__(self, models_dir: Path = ML_MODELS_DIR) -> None:
        self.metadata = load_json(models_dir / "best_model_info.json")
        self.model    = load_joblib(models_dir / "best_model.joblib")
        self.scaler   = load_joblib(models_dir / "scaler.joblib")
        ...

    def predict(self, payload: dict) -> dict:
        features_df = self._build_feature_frame(payload)
        x = self._apply_scaler(features_df)
        prediction = int(self.model.predict(x)[0])
        probability = self._predict_probability(x, prediction)
        return {
            "prediction": prediction,
            "label": CLASS_LABELS[prediction],
            "probability": probability,
            "model": self.model_key,
        }
```

#### Características

- **Aceita dois formatos de nome de feature** no payload:
  - nome original do dataset (`concave points_mean`)
  - snake_case amigável para APIs (`concave_points_mean`)
- **Aplica scaler condicionalmente** (apenas se `requires_scaler=True`)
- **Retorna probabilidade** quando o modelo suporta `predict_proba`
- **Validação robusta** → levanta `ValueError` listando features ausentes

---

## 🔄 Fluxo de execução

```
Modeling.modeling_results
       │
       ▼
select_best_model ──► escolhe modelo por (recall, f1, accuracy)
       │
       ▼
PersistBestModel ──► salva best_model.joblib + scaler.joblib + info.json
       │
       ▼
    (fim do pipeline de treino)

           ┌───────────── API Flask ─────────────┐
           │                                      │
           │   MLPredictor.__init__ (lazy load)   │
           │           │                          │
           │           ▼                          │
           │   POST /predict → predictor.predict()│
           │                                      │
           └──────────────────────────────────────┘
```

---

## 📦 Saída

Ao final do pipeline, a pasta `models/ml/` fica preparada para servir a API:

```
models/ml/
├── best_model.joblib
├── scaler.joblib
└── best_model_info.json
```

Esses arquivos são versionáveis, portáveis e independentes do pipeline de treino.

---

## ✅ Benefícios da abordagem

- **Desacoplamento** → treino e inferência são independentes; basta carregar os artefatos
- **Reprodutibilidade** → o `best_model_info.json` documenta exatamente qual modelo, com quais métricas e features foi persistido
- **Flexibilidade de nomes** → API aceita tanto `concave_points_mean` quanto `concave points_mean`
- **Segurança** → validação de features ausentes antes de chegar no modelo

---

## 📌 Conclusão

A camada de inference fecha o ciclo de Machine Learning do projeto: depois de treinar e comparar modelos, ela escolhe o melhor, o persiste em disco junto com o scaler e os metadados, e expõe uma interface simples (`MLPredictor`) que a API Flask utiliza para servir predições.

👉 É esse desenho que permite ao projeto sair do notebook de experimentação e chegar, de fato, a uma **API pronta para uso clínico**.

---

## ▶️ Execução

```bash
# Gera os artefatos (best_model.joblib, scaler.joblib, best_model_info.json)
python -m src.main

# Sobe a API que consome esses artefatos
python -m src.main --mode api
```
