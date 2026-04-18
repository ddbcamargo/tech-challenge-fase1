# 🧩 Pipeline

## 🎯 Objetivo

A etapa de pipeline tem como objetivo orquestrar todas as demais etapas do projeto (EDA, Preprocessing, Tuning, Modeling, Evaluation, Explainability e Inference) em uma sequência clara, previsível e fácil de manter.

O pipeline concentra a **ordem de execução** em um único lugar e utiliza um **contexto compartilhado** para transportar dados entre as etapas sem acoplar uma à outra.

---

## 🧱 Componentes

A camada de pipeline é composta por três peças:

### 🔹 `step.py` — `Step`

Classe abstrata que define o contrato de uma etapa do pipeline.

```python
class Step(ABC):
    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        ...
```

Toda etapa do projeto (EDA, Preprocessing, Tuning, etc.) herda de `Step` e implementa o método `execute`, que recebe o contexto, executa sua lógica e devolve o contexto atualizado.

👉 Esse contrato torna as etapas **intercambiáveis e desacopladas**: é possível adicionar, remover ou reordenar steps sem mexer nos outros.

---

### 🔹 `pipeline_context.py` — `PipelineContext`

`dataclass` que representa o estado compartilhado entre as etapas:

| Campo              | Descrição                                              |
| ------------------ | ------------------------------------------------------ |
| `df`               | DataFrame carregado e limpo na EDA                     |
| `data`             | Splits (train/test), dados escalados, scaler, features |
| `tuning_results`   | Melhor estimador de cada modelo após o GridSearchCV    |
| `modeling_results` | Métricas obtidas no conjunto de teste para cada modelo |
| `evaluation_df`    | DataFrame consolidado de métricas para comparação      |
| `best_model_info`  | Metadados do modelo vencedor (preenchido no final)     |

👉 O contexto funciona como uma "mochila" que percorre o pipeline: cada etapa lê o que precisa e adiciona o que produz.

---

### 🔹 `machine_learning_pipeline.py` — `MachineLearningPipeline`

Orquestrador principal. Define:

- **Quais etapas rodam** (lista de `Step`)
- **Em qual ordem**
- Executa cada etapa em sequência, encadeando o contexto

```python
class MachineLearningPipeline:
    def __init__(self) -> None:
        self.context = PipelineContext()
        self.steps: list[Step] = [
            ExploratoryDataAnalysis(),
            Preprocessing(),
            KNNTune(),            KNNModeling(),
            LogisticRegressionTune(),  LogisticRegressionModeling(),
            RandomForestTune(),        RandomForestModeling(),
            Evaluation(),
            ShapAnalysis(),
            PersistBestModel(),
        ]

    def run(self) -> PipelineContext:
        for step in self.steps:
            self.context = step.execute(self.context)
        return self.context
```

---

## 🔄 Fluxo de execução

```
EDA
 └─► Preprocessing
      └─► KNN Tune → KNN Modeling
      └─► Logistic Regression Tune → Logistic Regression Modeling
      └─► Random Forest Tune → Random Forest Modeling
           └─► Evaluation
                └─► SHAP Analysis
                     └─► PersistBestModel (modelo + scaler + metadados)
```

Cada etapa é independente: recebe o contexto, executa sua parte e devolve o contexto enriquecido para a próxima.

---

## ✅ Benefícios da abordagem

- **Desacoplamento** → cada etapa é uma classe isolada, testável e substituível
- **Legibilidade** → a ordem de execução fica visível em um único arquivo
- **Extensibilidade** → novas etapas (por exemplo, feature selection) podem ser inseridas sem reescrever o restante
- **Reprodutibilidade** → mesmo contexto de entrada → mesmo contexto de saída

---

## 📌 Conclusão

A camada de pipeline é a "espinha dorsal" do projeto. Ela garante que todas as etapas rodem na ordem correta, compartilhem estado de forma consistente e permaneçam independentes umas das outras — facilitando manutenção, evolução e depuração.

---

## ▶️ Execução

```bash
python -m src.main
```
