# 📈 Previsão de Preço de Ações com LinearSVC (scikit-learn)

Este repositório foi criado como parte da aula de **Fundamentos de Python e Introdução a Machine Learning e Inteligência Artificial**, do curso de **Pós-Graduação em Machine Learning Engineering** da **FIAP**.

## 🧠 Objetivo

Aplicar conceitos básicos de machine learning com Python utilizando a biblioteca `scikit-learn`, mais especificamente o modelo `LinearSVC`, para prever se o preço de uma ação vai **subir (1)** ou **cair (0)** com base em algumas características.

---

## 🧪 Sobre o Código

Neste projeto, simulamos um conjunto de dados com características simples de ações para treinar um classificador linear de suporte a vetores (`LinearSVC`). O código está dividido em etapas bem definidas:

### 🔹 1. Importações

```python
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
```

Importamos o modelo de classificação e uma métrica de avaliação (acurácia).

---

### 🔹 2. Dados de Treino

Cada ação é representada por três características (valores binários: 1 para sim, 0 para não):


```python
acao1 = [1, 0, 1] # AAPL
acao2 = [0, 1, 0] # GOOGL
acao3 = [1, 1, 1] # MSFT
acao4 = [0, 0, 1] # AMZN
acao5 = [1, 1, 0] # TSLA
acao6 = [0, 1, 1] # FB

dados_treino = [acao1, acao2, acao3, acao4, acao5, acao6]
rotulos_treino = [1, 1, 1, 0, 0, 0]
```

O vetor `rotulos_treino` contém o resultado real: **1** se o preço subiu, **0** se caiu.

---

### 🔹 3. Treinamento do Modelo

```python
modelo = LinearSVC()
modelo.fit(dados_treino, rotulos_treino)
```

Treinamos o modelo com os dados e rótulos definidos.

---

### 🔹 4. Teste do Modelo

```python
teste1 = [1, 0, 0]
teste2 = [0, 1, 1]
teste3 = [1, 1, 0]

dados_teste = [teste1, teste2, teste3]
rotulos_teste = [1, 0, 1]
```

Fornecemos um novo conjunto de dados de ações para avaliar se o modelo consegue prever corretamente.

---

### 🔹 5. Avaliação

```python
previsoes = modelo.predict(dados_teste)
taxa_acerto = accuracy_score(rotulos_teste, previsoes)
print("Taxa de acerto %.2f%%" % (taxa_acerto * 100))
```

Comparamos as previsões do modelo com os rótulos reais e calculamos a **taxa de acerto**.

---

## ✅ Resultado

O modelo imprime a taxa de acerto nas previsões, permitindo avaliar a performance do classificador com base nos dados simulados.

---

## 📚 Tecnologias Utilizadas

- Python 3.x
- scikit-learn (`LinearSVC`, `accuracy_score`)

---

## 📌 Observações

Este exemplo é simplificado e visa apenas demonstrar o funcionamento básico de um classificador supervisionado com dados binários. Em cenários reais de previsão de ações, o volume de dados, a complexidade dos atributos e o pré-processamento são significativamente maiores.

---
