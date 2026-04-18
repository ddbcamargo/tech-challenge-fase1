# Pré-processamento dos Dados

## Objetivo

A etapa de pré-processamento tem como objetivo preparar os dados para a modelagem, garantindo que estejam organizados e em formato adequado para os algoritmos de Machine Learning.

---

## Etapas realizadas

### 1. Separação entre variáveis de entrada e variável alvo

O dataset foi dividido em:

- `X`: variáveis preditoras
- `y`: variável alvo (`diagnosis`)

A variável `diagnosis` representa:

- 0 → Benigno
- 1 → Maligno

---

### 2. Separação entre treino e teste

Os dados foram divididos em:

- 80% para treino
- 20% para teste

Foi utilizado `stratify=y` para manter a proporção entre as classes nos dois conjuntos.

---

### 3. Escalonamento dos dados

Foi aplicado o `StandardScaler` para normalizar as variáveis numéricas.

Essa etapa é importante principalmente para algoritmos sensíveis à escala dos dados, como:

- KNN
- Regressão Logística

---

## Saídas geradas

A etapa de pré-processamento retorna:

- `x`
- `y`
- `x_train`
- `x_test`
- `y_train`
- `y_test`
- `x_train_scaled`
- `x_test_scaled`
- `scaler`

---

## Conclusão

Após o pré-processamento, os dados estão devidamente separados, balanceados entre treino e teste e normalizados para uso nos modelos de classificação.

---

## Execução

```bash
python -m src.main