# 📊 Análise Exploratória de Dados (EDA)

## 🎯 Objetivo

A etapa de Análise Exploratória de Dados (EDA - Exploratory Data Analysis) tem como objetivo compreender o dataset utilizado, identificar padrões, verificar a qualidade dos dados e preparar o terreno para as etapas de pré-processamento e modelagem.

---

## 📁 Dataset

- Nome: Breast Cancer Wisconsin Dataset  
- Problema: Classificação de tumores em benignos ou malignos  
- Tipo: Dados estruturados (tabulares)

A variável alvo é:

- `diagnosis`
  - 0 → Benigno
  - 1 → Maligno

---

## 🔧 Etapas realizadas

### 1. Carregamento dos dados

Os dados foram carregados a partir do arquivo "resources/data/breast_cancer.csv" utilizando a biblioteca `pandas`.

---

### 2. Limpeza dos dados

Foram realizadas as seguintes transformações:

- Remoção de colunas irrelevantes:
  - `id` (identificador sem valor preditivo)
  - `Unnamed: 32` (coluna vazia gerada pelo CSV)

- Conversão da variável alvo:
  - `"M"` → 1 (Maligno)
  - `"B"` → 0 (Benigno)

Essa transformação é necessária para permitir o uso de algoritmos de Machine Learning.

---

### 3. Análise estrutural

Foram avaliados:

- Dimensão do dataset  
- Tipos de dados  
- Estatísticas descritivas  
- Valores ausentes  

📊 Resultados:

- Total de registros: **569**
- Total de variáveis: **31**
- Não há valores nulos
- Todas as variáveis estão em formato numérico (float ou int)

👉 Conclusão: o dataset está limpo e adequado para modelagem.

---

### 4. Distribuição da variável alvo

Foi gerado um gráfico para analisar o balanceamento das classes.

📊 Resultados:

- Benigno (0): **357 casos**
- Maligno (1): **212 casos**

👉 Interpretação:

A base apresenta um leve desbalanceamento, com maior número de casos benignos.  
No entanto, o desbalanceamento não é crítico e não exige técnicas adicionais de balanceamento nesta etapa.

---

### 5. Análise de correlação

Foi gerado um mapa de correlação entre todas as variáveis numéricas.

📊 Principais observações:

- Forte correlação entre:
  - `radius`, `perimeter` e `area`
- Variáveis relacionadas ao tamanho do tumor apresentam alta relevância
- Existência de possíveis redundâncias entre variáveis

👉 Interpretação:

A alta correlação indica que algumas variáveis carregam informações semelhantes, o que pode influenciar modelos lineares e deve ser considerado na etapa de modelagem.

---

## 📈 Gráficos gerados

Os seguintes gráficos foram gerados e salvos em "resources/outputs/graphs" 
Arquivos:

- `target_distribution.png` → distribuição das classes  
- `correlation_heatmap.png` → mapa de correlação  

Esses gráficos auxiliam na interpretação dos dados e na construção do relatório.

---

## 📌 Conclusão

- O dataset está limpo e sem valores ausentes  
- A variável alvo está corretamente estruturada  
- Existe leve desbalanceamento entre as classes  
- Há forte correlação entre algumas variáveis  
- Os dados estão prontos para a etapa de pré-processamento  

---

## ▶️ Execução

```bash
python src/main.py