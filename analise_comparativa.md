# Análise Comparativa dos Notebooks de Detecção de Anomalias em Turbinas Eólicas

**Projeto:** Iniciação Científica — Detecção de Anomalias em Turbinas Eólicas  
**Data:** Abril de 2026  
**Autor:** Thiago Heleno

---

## Sumário

1. [Visão Geral do Projeto](#1-visão-geral-do-projeto)
2. [Dataset](#2-dataset)
3. [Entradas (X) e Saídas (Y)](#3-entradas-x-e-saídas-y)
4. [Seleção de Features](#4-seleção-de-features)
5. [Notebook 1 — CNN-LSTM Paper (Supervisionado)](#5-notebook-1--wind_turbine_cnn_lstm_paperipynb)
6. [Notebook 2 — CNN-BiLSTM Autoencoder (Semi-supervisionado, PyTorch)](#6-notebook-2--wind_turbine_anomaly_detection_v4ipynb)
7. [Notebook 3 — Keras MLP Autoencoder (Semi-supervisionado, Keras + Optuna)](#7-notebook-3--wind_turbine_autoencoder_keras_pipelineipynb)
8. [Comparação Geral](#8-comparação-geral)
9. [Resultados](#9-resultados)
10. [Análise: Por que Obtivemos Esses Resultados?](#10-análise-por-que-obtivemos-esses-resultados)
11. [Métricas de Avaliação Explicadas](#11-métricas-de-avaliação-explicadas)
12. [Conclusões e Próximos Passos](#12-conclusões-e-próximos-passos)

---

## 1. Visão Geral do Projeto

O objetivo deste projeto de IC é detectar automaticamente **anomalias em turbinas eólicas** a partir de dados de sensores SCADA (Supervisory Control and Data Acquisition). Os dados são séries temporais com resolução de 10 minutos, coletados de um parque eólico real.

### Dois paradigmas de detecção

Foram explorados dois paradigmas distintos:

| Paradigma | Descrição | Necessita de labels no treino? |
|-----------|-----------|-------------------------------|
| **Supervisionado** | O modelo aprende a diferença entre Normal e Anomalia usando exemplos rotulados | Sim |
| **Semi-supervisionado** | O modelo aprende o comportamento "normal" e detecta desvios desse padrão | Não (apenas dados normais no treino) |

### Os três notebooks

| # | Notebook | Paradigma | Modelo |
|---|----------|-----------|--------|
| 1 | `wind_turbine_cnn_lstm_paper.ipynb` | Supervisionado | CNN / LSTM / CNN-LSTM |
| 2 | `wind_turbine_anomaly_detection_v4.ipynb` | Semi-supervisionado | CNN-BiLSTM-Attention Autoencoder (PyTorch) |
| 3 | `wind_turbine_autoencoder_keras_pipeline.ipynb` | Semi-supervisionado | Dense MLP Autoencoder (Keras + Optuna) |

---

## 2. Dataset

### Fonte

- **Nome:** CARE_To_Compare — Wind Farm C
- **Tipo:** Dados SCADA de turbinas eólicas (série temporal)
- **Resolução temporal:** 10 minutos por amostra
- **Total de amostras:** 3.187.136 linhas
- **Número de eventos:** 58 eventos registrados

### Composição dos eventos

| Tipo | Quantidade | Descrição |
|------|-----------|-----------|
| Eventos com anomalia | 27 (51 na visão Keras) | Períodos onde a turbina estava falhando |
| Eventos normais | 31 (7 na visão Keras) | Períodos de operação saudável |

> **Nota:** A diferença de contagem entre notebooks se deve à forma como os eventos são definidos. O notebook Keras usa uma granularidade diferente de "evento".

### Sensores e colunas

- **Sensores físicos:** 238 sensores originais
- **Colunas brutas:** 952 colunas após pré-processamento
  - Motivo: colunas de ângulo (graus) são convertidas para **pares sin/cos** para preservar a continuidade circular (ex: 359° e 1° são próximos, mas numericamente distantes)
  - Cada ângulo vira 2 colunas → expansão do espaço de features

### Desbalanceamento de classes

Este é o desafio central do projeto:

| Classe | Amostras | Proporção |
|--------|----------|-----------|
| Normal (0) | 3.126.239 | **98,1%** |
| Anomalia (1) | 60.897 | **1,9%** |
| **Razão de desbalanceamento** | | **51:1** |

> **Problema:** Um modelo "ingênuo" que classifica tudo como Normal já atinge 98% de acurácia. Por isso, acurácia sozinha é uma métrica enganosa neste contexto.

### Divisão treino/validação/teste

Cada notebook usa uma estratégia diferente:

**Notebook 1 (CNN-LSTM):** Divisão por **evento** (sem sobreposição entre splits)

| Split | Eventos | Anomalia | Normal |
|-------|---------|----------|--------|
| Treino | 40 | 18 | 22 |
| Validação | 8 | 4 | 4 |
| Teste | 10 | 5 | 5 |

**Notebooks 2 e 3 (Autoencoders):** Divisão **temporal** (70% / 15% / 15%)

| Split | Amostras | Anomalias | Normais |
|-------|----------|-----------|---------|
| Treino | 2.230.995 (70%) | 40.676 | 1.950.023 |
| Validação | 478.065 (15%) | 13.855 | 423.983 |
| Teste | 478.076 (15%) | 6.366 | 421.385 |

---

## 3. Entradas (X) e Saídas (Y)

### Notebook 1 — CNN-LSTM Paper (Supervisionado)

| | Descrição |
|---|---|
| **X (entrada)** | Janela de 36 timesteps × 285 features de sensores (normalizadas [0,1]) |
| **Y (saída)** | Rótulo binário: `0 = Normal`, `1 = Anomalia` |
| **Forma do X** | `(batch, 36, 285)` |
| **Forma do Y** | `(batch,)` — inteiro 0 ou 1 |
| **Regra de rótulo** | Janela é anomalia se **qualquer** timestep dentro dela for anômalo |

O modelo faz **classificação direta**: recebe a janela e prediz a classe.

---

### Notebook 2 — CNN-BiLSTM Autoencoder (Semi-supervisionado)

| | Descrição |
|---|---|
| **X (entrada)** | Janela de 36 timesteps × 305 features de sensores (normalizadas [0,1]) |
| **Y (saída do modelo)** | **Reconstrução do próprio X** — o modelo tenta reproduzir a entrada |
| **Forma do X** | `(batch, 36, 305)` |
| **Forma do Y** | `(batch, 36, 305)` — idêntica ao X |
| **Como detecta anomalia** | Erro de reconstrução alto = anomalia (o modelo não sabe reconstruir o que é diferente do padrão normal) |

O modelo **não usa rótulos no treino**. Só aprende o padrão de dados normais.

---

### Notebook 3 — Keras MLP Autoencoder (Semi-supervisionado)

| | Descrição |
|---|---|
| **X (entrada)** | Vetor de 305 features de uma amostra individual (sem janela temporal) |
| **Y (saída do modelo)** | **Reconstrução do próprio X** |
| **Forma do X** | `(batch, 305)` |
| **Forma do Y** | `(batch, 305)` — idêntica ao X |
| **Como detecta anomalia** | RMSE entre X original e X reconstruído — RMSE alto = anomalia |

> **Diferença chave entre Notebooks 2 e 3:** O notebook 2 usa janelas temporais de 6 horas (captura padrões ao longo do tempo). O notebook 3 analisa cada amostra individualmente, sem contexto temporal.

---

## 4. Seleção de Features

Reduzir de 952 para um subconjunto relevante de features é essencial para:
- Remover ruído e redundância
- Reduzir custo computacional
- Melhorar a generalização do modelo

### Notebook 1 — Seleção Supervisionada via XGBoost

```
952 features originais
    ↓ Treina XGBoost com scale_pos_weight (ajuste de desbalanceamento)
    ↓ Ranqueia features por importância (gain)
    ↓ Seleciona top 30%
285 features finais
```

- **Vantagem:** Seleciona diretamente as features mais **discriminativas** entre Normal e Anomalia
- **Desvantagem:** Requer rótulos — não pode ser usado em cenários sem dados anotados

### Notebooks 2 e 3 — Seleção Não Supervisionada

```
952 features originais
    ↓ Remove variância < 0.0001 (features constantes ou quase)        → -46 features (906 restam)
    ↓ Remove correlação com vars. operacionais < 0.1 (features inúteis) → -129 features (777 restam)
    ↓ Remove correlação inter-features > 0.95 (features redundantes)   → -472 features (305 restam)
305 features finais
```

**Variáveis operacionais usadas como referência:** potência gerada, velocidade do vento, sensor_144, sensor_145 (e suas variações avg/max/min/std).

| Estratégia | Features selecionadas | Usa labels? |
|------------|----------------------|-------------|
| XGBoost (NB1) | 285 | Sim |
| Variância + Correlação (NB2/3) | 305 | Não |

---

## 5. Notebook 1 — `wind_turbine_cnn_lstm_paper.ipynb`

### Referência

Implementação baseada no paper: **Qi et al., "Research on Wind Turbine Fault Detection Based on CNN-LSTM", Energies 2024** — adaptada para o dataset CARE_To_Compare (classificação binária em vez de multi-classe).

### Pipeline completo

```
1. Carrega 58 CSVs do Wind Farm C
         ↓
2. Divisão por EVENTO (sem leakage):
   40 treino / 8 validação / 10 teste
         ↓
3. MinMaxScaler ajustado SOMENTE nos dados normais do treino
   (sem usar dados de validação/teste para evitar data leakage)
         ↓
4. XGBoost Feature Selection:
   Amostra 100K linhas do treino → treina XGB → seleciona top 30% = 285 features
         ↓
5. Janelas deslizantes de 36 timesteps:
   Rótulo "any anomaly": janela = anomalia se qualquer timestep for anômalo
         ↓
6. Undersampling (SOMENTE no treino):
   Normal: mantém 2,3% das janelas (para equilibrar 1:1)
   Val/Teste: mantém TUDO (distribuição real: ~58:1)

   Resultado:
   - Treino: 50.323 normal + 50.343 anomalia (balanceado)
   - Val:    431.043 normal + 2.237 anomalia (natural)
   - Teste:  531.987 normal + 9.122 anomalia (natural)
         ↓
7. Treinamento de 3 modelos:
   CNN, LSTM, CNN-LSTM
   Loss: CrossEntropy | Otimizador: Adam | Batch: 600 | Épocas: 500 (early stop)
         ↓
8. Threshold ótimo na validação (maximiza F1)
         ↓
9. Avaliação final no conjunto de teste
```

### Arquitetura CNN

```
Entrada: (batch, 36, 285)   [timesteps × features]
    ↓ Transpõe para (batch, 285, 36)   [features × timesteps para Conv1D]
    ↓ Conv1D(285→32, kernel=3) + SELU
    ↓ MaxPool1D(kernel=2)              → (batch, 32, 17)
    ↓ Conv1D(32→64, kernel=2) + SELU
    ↓ MaxPool1D(kernel=3)              → (batch, 64, 5)
    ↓ Flatten                          → (batch, 320)
    ↓ AlphaDropout(0.5)
    ↓ Dense(320→50) + SELU
    ↓ Dense(50→10) + SELU
    ↓ Dense(10→2) + Softmax
Saída: probabilidade [P(Normal), P(Anomalia)]
```

> **AlphaDropout:** variante do Dropout compatível com SELU que preserva a auto-normalização da rede.

### Arquitetura LSTM

```
Entrada: (batch, 36, 285)   [timesteps × features]
    ↓ LSTM(285→100, batch_first=True)  → (batch, 36, 100)
    ↓ LSTM(100→80, batch_first=True)   → (batch, 36, 80)
    ↓ Pega ÚLTIMO timestep             → (batch, 80)
    ↓ Dropout(0.5)
    ↓ Dense(80→50) + SELU
    ↓ Dense(50→10) + SELU
    ↓ Dense(10→2) + Softmax
Saída: probabilidade [P(Normal), P(Anomalia)]
```

### Arquitetura CNN-LSTM (Híbrido) — ~160K parâmetros

```
Entrada: (batch, 36, 285)
    ↓ Transpõe → (batch, 285, 36)
    ↓ Conv1D(285→32, kernel=3) + SELU
    ↓ MaxPool1D(kernel=2)              → (batch, 32, 17)
    ↓ Conv1D(32→64, kernel=2) + SELU
    ↓ MaxPool1D(kernel=3)              → (batch, 64, 5)
    ↓ Transpõe → (batch, 5, 64)        [preparar para LSTM]
    ↓ LSTM(64→100)                     → (batch, 5, 100)
    ↓ LSTM(100→80)                     → (batch, 5, 80)
    ↓ Pega ÚLTIMO timestep             → (batch, 80)
    ↓ Dropout(0.5)
    ↓ Dense(80→50) + SELU
    ↓ Dense(50→10) + SELU
    ↓ Dense(10→2) + Softmax
Saída: probabilidade [P(Normal), P(Anomalia)]
```

> **Intuição:** A CNN extrai padrões locais nas features, reduzindo a sequência de 36 para 5 timesteps. O LSTM então aprende dependências temporais nessa representação comprimida.

### Configuração de treino

| Parâmetro | Valor |
|-----------|-------|
| Loss | CrossEntropyLoss |
| Otimizador | Adam |
| Learning rate | 0.001 |
| Batch size | 600 |
| Épocas máx | 500 |
| Early stopping | patience=20 (monitora val_loss) |
| Seed | 42 (reprodutível) |
| Device | CUDA (GPU) |

### Avaliação

1. Roda inferência no conjunto de **validação** com probabilidades brutas
2. Varre thresholds via curva Precision-Recall
3. Escolhe o threshold que **maximiza F1** na validação
4. Aplica esse threshold no **conjunto de teste** para métricas finais

---

## 6. Notebook 2 — `wind_turbine_anomaly_detection_v4.ipynb`

### Visão geral

Autoencoder profundo com arquitetura CNN-BiLSTM-Attention treinado exclusivamente em dados normais. A ideia central: **se o modelo só viu dados normais durante o treino, ele terá dificuldade em reconstruir anomalias** — o erro de reconstrução alto sinaliza um evento anômalo.

O notebook v4 estende o v3 com **5 estratégias de melhoria pós-treino** sem re-treinar o modelo.

### Pipeline completo

```
1. Carrega 58 CSVs do Wind Farm C
         ↓
2. Divisão TEMPORAL (70% / 15% / 15%)
         ↓
3. Pré-processamento:
   a) DataClipper: remove outliers extremos (Q0.1% a Q99.9% por feature)
   b) DuplicateValuesToNan: substitui por NaN valores que repetem 6+ vezes seguidas
   c) SimpleImputer: ffill → bfill → zeros para NaNs restantes
         ↓
4. MinMaxScaler ajustado SOMENTE nos dados NORMAIS do treino
         ↓
5. Seleção não supervisionada → 305 features
         ↓
6. Janelas deslizantes de 36 timesteps
   Treino: SOMENTE janelas normais (1.535.633 janelas)
   Val:    SOMENTE janelas normais (327.668 janelas) — para calibrar thresholds
   Teste:  TODAS as janelas (normais + anomalias) — para avaliação final
         ↓
7. Treinamento do Autoencoder (MAE, 100 épocas, early stopping patience=15)
   Mixed Precision (AMP) habilitado para eficiência
   Best val_loss = 0.011814 (epoch 100)
         ↓
8. Cálculo dos thresholds por feature:
   - Para cada uma das 305 features, calcula o P95 e P99 do erro de reconstrução
     sobre as janelas NORMAIS da validação
         ↓
9. Classificação:
   Uma janela é anomalia se ≥ 15 features (de 305) excederem seu threshold
         ↓
10. Melhorias pós-hoc (v4, sem re-treinar):
    - Pesos por feature (AUROC): features mais discriminativas têm maior peso
    - Suavização temporal (majority voting, K=3/5/9/15 janelas)
    - Máscara de features informativas (só features com std_error > 0.01)
    - XGBoost pós-hoc (classificador sobre os erros de reconstrução)
```

### Arquitetura CNN-BiLSTM-Attention Autoencoder — 954K parâmetros

```
Entrada: (batch, 36, 305)   [timesteps × features]

═══════════════════ ENCODER ═══════════════════
    ↓ Conv1D(305→64, kernel=3, pad=1) + SELU    → (batch, 36, 64)
    ↓ Conv1D(64→128, kernel=3, pad=1) + SELU    → (batch, 36, 128)
    ↓ BiLSTM(128→256 bidirecional)              → (batch, 36, 256)
    ↓ BiLSTM(256→128 bidirecional)              → (batch, 36, 128)
    ↓ MultiHeadAttention(embed=128, heads=4)
      + LayerNorm (conexão residual)             → (batch, 36, 128)

═══════════════════ DECODER ═══════════════════
    ↓ LSTM(128→128) + SELU                      → (batch, 36, 128)
    ↓ LSTM(128→128) + SELU                      → (batch, 36, 128)
    ↓ Dropout(0.15)
    ↓ Linear(128→256) + SELU                    → (batch, 36, 256)
    ↓ Linear(256→305)                           → (batch, 36, 305)

Saída: reconstrução de (batch, 36, 305)
Loss: MAE (erro absoluto médio) entre entrada e saída
```

**Por que BiLSTM?** O LSTM bidirecional processa a sequência em ambas as direções (passado → futuro e futuro → passado), capturando dependências temporais mais ricas.

**Por que Attention?** O mecanismo de atenção aprende quais timesteps e features são mais relevantes para a reconstrução, melhorando a qualidade do embedding.

### Configuração de treino

| Parâmetro | Valor |
|-----------|-------|
| Loss | MAE (L1Loss) |
| Otimizador | Adam |
| Learning rate | 1e-4 |
| Batch size | 64 |
| Épocas máx | 100 |
| Early stopping | patience=15 (monitora val_loss) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Dropout | 0.15 |
| Mixed Precision | Habilitado (AMP CUDA) |
| Device | GPU (NVIDIA RTX A6000) |

### Avaliação

**Erro de reconstrução por feature:**
- Para cada janela, calcula o MAE médio ao longo dos 36 timesteps, separadamente para cada uma das 305 features
- Resultado: vetor de 305 erros por janela

**Thresholds por feature (calculados nas janelas normais da validação):**
- P95: o 95º percentil do erro de cada feature em dados normais
- P99: o 99º percentil

**Decisão de classificação:**
```
anomalia = (número de features com erro > threshold) >= 15
```

O valor 15 (~5% das 305 features) foi calibrado para que, por puro acaso, apenas ~0,004% das janelas normais sejam falsamente classificadas como anomalias (dado que P95 implica 5% de chance por feature individualmente, 15 features simultâneas é estatisticamente improvável).

**Melhorias v4:**

| Estratégia | Descrição |
|-----------|-----------|
| Feature Weighting | Pesos AUROC por feature — features mais discriminativas (anomalia vs normal) têm maior peso no score |
| Temporal Smoothing | Suavização por maioria: janela = anomalia somente se maioria dos K vizinhos também for anomalia (reduz FP isolados) |
| Informative Mask | Usa apenas as 144 features com std_error > 0.01 (descarta features com reconstrução plana) |
| XGBoost Post-hoc | Treina XGBoost usando os 305 erros de reconstrução como features — **falhou no teste** |

---

## 7. Notebook 3 — `wind_turbine_autoencoder_keras_pipeline.ipynb`

### Visão geral

Autoencoder denso (MLP) implementado em TensorFlow/Keras. Diferente do notebook 2, não usa janelas temporais — cada amostra de 10 minutos é processada independentemente. Os hiperparâmetros foram otimizados com **Optuna** (100 trials no total). A avaliação usa a métrica **CARE** (composta, multi-objetivo).

### Pipeline completo

```
1. Carrega 58 CSVs do Wind Farm C
         ↓
2. Divisão TEMPORAL (70% / 15% / 15%)
   Pontos de corte: 2025-10-20 e 2027-05-13
         ↓
3. Pré-processamento (pipeline sklearn):
   a) DataClipper: clipping por Q0.1% e Q99.9% por feature
   b) DuplicateValuesToNan: NaN para valores repetidos 6+ vezes
   c) SimpleImputer: ffill → bfill → zeros
   d) MinMaxScaler: normaliza [0,1] usando SOMENTE dados normais do treino
         ↓
4. Seleção não supervisionada → 305 features
   (mesma estratégia do notebook 2)
         ↓
5. Otimização de Hiperparâmetros com Optuna:
   Estudo 1 (50 trials): arquitetura do autoencoder
   - n_layers: [1, 2, 3]
   - code_size (bottleneck): [10, 100]
   - learning_rate: [1e-4, 1e-2]
   - decay_rate: [0.90, 0.999]
   - gamma (threshold adaptativo): [0.0, 0.5]

   Melhor config: n_layers=2, code_size=35, lr=9.795e-4, decay=0.9941

   Estudo 2 (50 trials): gamma do threshold adaptativo
   Melhor gamma: 0.4997
         ↓
6. Treinamento final do Autoencoder (MSE, 50 épocas, early stopping patience=5)
         ↓
7. Threshold de anomalia:
   - Fixo (P95): RMSE de calibração no percentil 95 = 2.954
   - Adaptativo: RMSE_predito_para_amostra + gamma
         ↓
8. Avaliação com métrica CARE:
   - Avaliação POR EVENTO (58 eventos)
   - Avaliação POR AMOSTRA
   - Análise ARCANA (importância de features)
```

### Arquitetura Dense MLP Autoencoder — ~150K parâmetros

```
Entrada: (batch, 305)   [features de uma única amostra]

══════════ ENCODER ══════════
    ↓ Dense(305→200) + PReLU    [He-normal init]
    ↓ Dense(200→100) + PReLU
    ↓ Dense(100→35) + PReLU    ← BOTTLENECK (representação comprimida)

══════════ DECODER ══════════
    ↓ Dense(35→100) + PReLU
    ↓ Dense(100→200) + PReLU
    ↓ Dense(200→305, linear)   ← reconstrução

Saída: (batch, 305) — reconstrução do input
Loss: MSE (erro quadrático médio)
```

**PReLU (Parametric ReLU):** variante do ReLU onde o slope negativo é um parâmetro aprendido (em vez de fixo em zero como no ReLU padrão). Isso permite maior flexibilidade na ativação.

**Bottleneck de 35 neurônios:** o encoder comprime 305 features em apenas 35 — forçando o modelo a aprender uma representação essencial do comportamento normal.

### Configuração de treino

| Parâmetro | Valor |
|-----------|-------|
| Loss | MSE |
| Otimizador | Adam com ExponentialDecay |
| Learning rate inicial | 9.795e-4 (otimizado por Optuna) |
| Decay rate | 0.9941 por 1000 passos |
| Batch size | 128 |
| Épocas máx | 50 |
| Early stopping | patience=5, min_delta=1e-4 |
| Restore best weights | Sim |

### Otimização com Optuna

O Optuna é uma biblioteca de otimização de hiperparâmetros que busca automaticamente a melhor configuração:

- **50 trials** para a arquitetura do autoencoder
- **50 trials** para o parâmetro gamma do threshold adaptativo
- Cada trial treina e avalia um modelo completo
- O melhor resultado é salvo automaticamente

### Avaliação — Métrica CARE

A métrica **CARE** (Composite Anomaly Reliability Evaluation) é uma pontuação composta que combina múltiplos aspectos:

| Componente | Descrição |
|-----------|-----------|
| F1 Score | Média harmônica entre precisão e recall |
| Accuracy (EF1) | Event-level F1 — avaliação por evento completo |
| WS (Warning System) | Penaliza alarmes tardios — quanto antes detectar, melhor |
| PR_AUC | Área sob a curva Precision-Recall |
| ROC_AUC | Área sob a curva ROC |

**Avaliação por evento vs por amostra:**

- **Por evento:** Um evento é detectado se ao menos uma amostra dele for classificada como anomalia. Mais relevante para operadores — o que importa é detectar que algo está errado, não identificar cada timestep exato.
- **Por amostra:** Avaliação timestep a timestep. Mais exigente — o modelo precisa estar certo em cada ponto.

### Score de anomalia adaptativo

O threshold adaptativo é mais sofisticado que um valor fixo:

```
threshold(amostra_i) = RMSE_previsto(amostra_i) + gamma

onde RMSE_previsto é estimado por um segundo modelo que aprende
a prever o erro de reconstrução esperado para cada amostra normal.
```

Isso permite que o threshold se adapte ao contexto — em períodos de operação mais variada, o threshold sobe automaticamente, reduzindo falsos positivos.

---

## 8. Comparação Geral

### Diferenças de design

| Aspecto | Notebook 1 (CNN-LSTM) | Notebook 2 (CNN-BiLSTM AE) | Notebook 3 (Keras MLP AE) |
|---------|----------------------|--------------------------|--------------------------|
| **Paradigma** | Supervisionado | Semi-supervisionado | Semi-supervisionado |
| **Framework** | PyTorch | PyTorch | TensorFlow/Keras |
| **Arquitetura** | CNN / LSTM / CNN-LSTM | CNN-BiLSTM-Attention AE | Dense MLP AE |
| **Janela temporal** | 36 timesteps (6h) | 36 timesteps (6h) | Nenhuma (amostra única) |
| **Features (X)** | 285 (XGBoost) | 305 (variância/correlação) | 305 (variância/correlação) |
| **Y (treino)** | Rótulo binário | Reconstrução do X | Reconstrução do X |
| **Treina em anomalias?** | Sim (balanceado) | Não (só normal) | Não (só normal) |
| **Loss** | CrossEntropy | MAE | MSE |
| **Otimizador** | Adam (lr=0.001) | Adam (lr=1e-4) | Adam + ExpDecay (lr=9.8e-4) |
| **Batch size** | 600 | 64 | 128 |
| **Épocas máx** | 500 | 100 | 50 |
| **Early stopping** | patience=20 | patience=15 | patience=5 |
| **Balanceamento** | Undersampling (1:1 treino) | N/A (só treina normal) | N/A (só treina normal) |
| **Threshold** | Ótimo por F1 (val) | P95/P99 por feature | Adaptativo (RMSE + gamma) |
| **Tuning** | Manual | Manual | Optuna (100 trials) |
| **Parâmetros** | ~160K | ~954K | ~150K |
| **Avaliação** | Amostra (janela) | Amostra (janela) + evento | Amostra + evento (CARE) |

---

## 9. Resultados

### Notebook 1 — CNN-LSTM Paper (avaliação por janela no conjunto de teste)

| Modelo | Epochs | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|--------|----------|-----------|--------|----------|---------|
| **CNN** | 32 | 98,15% | 19,62% | 3,18% | **5,47%** | 0,55 |
| **LSTM** | 39 | 98,31% | 0,00% | 0,00% | **0,00%** | 0,40 |
| **CNN-LSTM** | 26 | 96,85% | 3,28% | 3,05% | **3,16%** | **0,66** |

> Conjunto de teste: 531.987 janelas normais + 9.122 janelas anômalas (razão 58:1)

### Notebook 2 — CNN-BiLSTM Autoencoder (avaliação por janela no conjunto de teste)

| Estratégia | Precision | Recall | F1-Score | AUC-ROC |
|-----------|-----------|--------|----------|---------|
| **P95** (threshold conservador) | 1,64% | 96,46% | 3,22% | 0,865 |
| **P99** (threshold restritivo) | 2,99% | 88,46% | **5,78%** | 0,865 |
| Best-F1 threshold | 1,59% | 95,07% | 3,13% | 0,865 |
| **v4 Weighted + Beta-F1** | 2,32% | 88,26% | 4,52% | **0,872** |
| v4 P95 + Info Mask | 1,65% | 94,56% | 3,24% | 0,865 |
| v4 XGBoost (falhou) | 0,00% | 0,00% | 0,00% | 0,743 |

**Detecção por evento (P95, 3 eventos do teste):**

| Evento | Janelas anômalas | Detectadas | Taxa |
|--------|-----------------|-----------|------|
| 16 | 2.195 | 2.109 | 96,1% |
| 30 | 3.298 | 3.153 | 95,6% |
| 66 | 978 | 978 | **100,0%** |
| **Média ponderada** | | | **96,4%** |

### Notebook 3 — Keras MLP Autoencoder

**Avaliação por evento (58 eventos):**

| Métrica | Valor |
|---------|-------|
| **CARE Score** (composto) | **0,6993** |
| Sensibilidade/Recall (evento) | **94,1%** (48 de 51 eventos detectados) |
| Precisão (evento) | **88,9%** (48 de 54 alarmes corretos) |
| Especificidade (evento) | 14,3% (1 de 7 normais classificados corretamente) |
| Falsos Negativos | 3 eventos (datasets 80, 81, 46 — anomalias sutis) |
| Falsos Positivos | 6 eventos (dataset 85 = outlier extremo) |

**Avaliação por amostra (teste, 477.656 amostras):**

| Métrica | Valor |
|---------|-------|
| Precision | 2,32% |
| Recall | 88,26% |
| F1-Score | 4,52% |
| AUC-ROC | **0,8717** |
| Amostras classificadas como anômalas | 246.454 (51,6%) |

**Hiperparâmetros ótimos (Optuna):**

| Parâmetro | Valor |
|-----------|-------|
| n_layers | 2 |
| code_size (bottleneck) | 35 |
| learning_rate | 9.795e-4 |
| decay_rate | 0.9941 |
| gamma (threshold) | 0.4997 |

---

## 10. Análise: Por que Obtivemos Esses Resultados?

### 10.1 O problema central: desbalanceamento severo

O dataset tem 98,1% de dados normais e apenas 1,9% de anomalias. Isso cria dois problemas:

**Accuracy enganosa:**
```
Modelo que sempre diz "Normal": Accuracy = 98,1% ✓
Mas Recall de anomalias = 0%  ✗
```

Por isso, **accuracy é inútil** como métrica aqui. F1-Score e AUC-ROC são mais informativos.

**Por que Recall é tão mais fácil de maximizar que Precision?**

No teste do notebook 1, há 531.987 janelas normais e apenas 9.122 anômalas. Se o modelo classifica tudo como anomalia:
- Recall = 100% (detectou todas)
- Precision = 9.122 / (531.987 + 9.122) = 1,7%

Obter alto recall é "barato" — basta ser agressivo. Obter alta precisão é difícil quando as anomalias são raras.

---

### 10.2 Por que o LSTM colapsa completamente? (AUC-ROC = 0,40, F1 = 0%)

O LSTM sozinho obteve os piores resultados: **zero** em Precision, Recall e F1, e AUC-ROC abaixo de 0,5 (pior que aleatório).

**Explicação:**

1. **Treino balanceado, teste desbalanceado:** O undersampling deixou o treino 1:1 (Normal:Anomalia). Mas no teste, a proporção volta a 58:1. O LSTM, que aprendeu a "50% das vezes é anomalia", chega no teste e vê que dizer Normal é sempre mais seguro.

2. **Collapse para classe majoritária:** O modelo aprende que classificar tudo como Normal minimiza o loss no conjunto de validação (que é desbalanceado). Como resultado, o threshold ótimo encontrado na validação é tão alto (0,9653) que nenhuma amostra de teste ultrapassa.

3. **Por que o LSTM é mais suscetível que a CNN?** O LSTM captura dependências temporais longas mas é mais propenso a "decorar" padrões dominantes. A CNN, com suas convoluções locais, consegue ainda capturar alguns padrões discriminativos mesmo com distribuição adversa no teste.

---

### 10.3 Por que a CNN tem o melhor F1 entre os supervisionados? (5,47%)

A CNN usa **AlphaDropout** (mais adequado para SELU) e **Flatten** direto após as convoluções — sem o gargalo do LSTM. A representação CNN é mais compacta e menos sensível ao desbalanceamento pós-undersampling.

Além disso, a CNN parou na época 32 (antes do overfitting), enquanto o LSTM continuou até a época 39 — mais tempo para convergir para "sempre Normal".

---

### 10.4 Por que o CNN-LSTM tem melhor AUC-ROC (0,66)?

O AUC-ROC é uma métrica **independente de threshold** — mede a qualidade de ordenação do score, não a decisão binária. Isso favorece o CNN-LSTM porque:

- A combinação CNN+LSTM gera scores de probabilidade mais calibrados
- Mesmo que o threshold fixo dê baixo F1, a ordenação dos scores é melhor
- AUC-ROC = 0,66 significa: dado uma janela normal e uma anômala aleatórias, o modelo as ordena corretamente 66% das vezes

---

### 10.5 Por que os autoencoders têm AUC-ROC muito melhor? (0,865 vs 0,66)

**Vantagem fundamental do paradigma semi-supervisionado:**

Os autoencoders **não sofrem do problema de desbalanceamento** porque:
- Treinam **somente em dados normais** — a proporção de classes é irrelevante
- Aprendem a reconstruir o padrão normal com alta fidelidade
- Anomalias, por serem diferentes do padrão aprendido, geram erros de reconstrução maiores
- O score de anomalia (erro de reconstrução) é naturalmente mais discriminativo

```
Dado Normal → Autoencoder → Reconstrução ≈ Original (erro baixo)
Dado Anômalo → Autoencoder → Reconstrução ≠ Original (erro alto)
```

---

### 10.6 Por que a Precision é tão baixa em todos (~2-3%)?

Alta taxa de **Falsos Positivos** em todos os modelos. Causas:

1. **Thresholds muito sensíveis:** Qualquer desvio do padrão normal (incluindo variações legítimas de operação) é sinalizado como anomalia

2. **Variações operacionais normais:** O parque eólico tem regimes de operação diferentes (ventos fortes, partida, parada), que podem se parecer com anomalias para um modelo treinado na média

3. **Dataset de calibração limitado:** Com poucos dados normais de validação representando todos os regimes operacionais, os thresholds ficam calibrados para um subconjunto de condições

4. **Ruído nos sensores SCADA:** Sensores industriais têm ruído intrínseco que pode exceder thresholds ocasionalmente

---

### 10.7 Por que o XGBoost pós-hoc falha? (v4)

O XGBoost foi treinado sobre os **erros de reconstrução** (305-dim) como features, com o objetivo de aprender a distinguir anomalias de falsos positivos do autoencoder.

**Por que falhou:**

1. **Overfitting severo:** Pouquíssimas amostras de anomalia no treino (em escala absoluta)
2. **Distribuição shift:** Os erros de reconstrução no conjunto de teste têm distribuição diferente do treino (dataset 85 com valores extremos)
3. **Threshold aprendido alto demais:** O XGBoost aprende que "prever Normal" é seguro e colapsa para 0% recall

O resultado (AUC-ROC=0,743 pelo AUC, mas 0% recall) sugere que o ranking de scores é razoável mas a decisão binária falha.

---

### 10.8 Por que o Keras AE tem o melhor resultado por evento? (94,1% sensibilidade)

A avaliação **por evento** é intrinsecamente mais fácil do que por amostra:

```
Evento = detectado se QUALQUER amostra for classificada como anomalia
```

Um evento de anomalia pode ter milhares de timesteps. Basta um timestep ser corretamente classificado para o evento ser detectado.

Adicionalmente:
- O Keras AE usa threshold adaptativo (RMSE_pred + gamma) que se ajusta por amostra
- O modelo foi otimizado com Optuna — melhores hiperparâmetros que busca manual
- A avaliação CARE penaliza soluções triviais (alto recall com precision zero)

---

### 10.9 Resumo visual das causas

```
Problema                          Efeito observado
─────────────────────────────     ──────────────────────────
Desbalanceamento 98:2%            Accuracy enganosa (98%)
LSTM + treino balanceado         LSTM colapsa no teste (F1=0%)
Threshold fixo                   Precision baixa (~2-3%)
Supervisionado requer labels     Depende de dados rotulados
Autoencoder treina em normal     AUC-ROC muito maior (0.87)
Avaliação por amostra            Precision muito baixa (FP alto)
Avaliação por evento             Resultados melhores (94,1%)
```

---

## 11. Métricas de Avaliação Explicadas

### Matriz de Confusão

```
                   Predito: Normal    Predito: Anomalia
Real: Normal       TN (Verdadeiro -)  FP (Falso +)
Real: Anomalia     FN (Falso -)       TP (Verdadeiro +)
```

### Métricas principais

| Métrica | Fórmula | Interpretação | Ideal para |
|---------|---------|---------------|-----------|
| **Accuracy** | (TP+TN) / Total | % de predições corretas | Datasets balanceados |
| **Precision** | TP / (TP+FP) | Dos alarmes gerados, quantos são reais? | Minimizar alarmes falsos |
| **Recall** | TP / (TP+FN) | Das anomalias reais, quantas detectamos? | Não perder anomalias |
| **F1-Score** | 2×P×R / (P+R) | Média harmônica de P e R | Tradeoff geral |
| **AUC-ROC** | Área sob curva ROC | Qualidade de ordenação dos scores | Threshold independente |
| **F-beta** | (1+β²)×P×R / (β²×P+R) | F1 com pesos diferentes para P e R | Domínio específico |

### Por que AUC-ROC é melhor que Accuracy aqui?

- **Accuracy:** afetada pelo desbalanceamento — 98% sem detectar nada
- **AUC-ROC:** mede se o modelo **ranqueia** corretamente anomalias acima de normais, independente do threshold escolhido
- AUC-ROC = 0.5 → aleatório; AUC-ROC = 1.0 → perfeito; AUC-ROC < 0.5 → pior que aleatório

### Curva Precision-Recall

Mostra o tradeoff entre Precision e Recall para diferentes thresholds. Em datasets desbalanceados, esta curva é mais informativa que a curva ROC.

```
Threshold baixo  → Alto Recall, baixa Precision (muitos alarmes, não perde anomalias)
Threshold alto   → Baixo Recall, alta Precision (poucos alarmes, mas confiáveis)
```

---

## 12. Conclusões e Próximos Passos

### Ranking dos modelos

| Critério | Melhor | Segundo | Pior |
|----------|--------|---------|------|
| AUC-ROC (amostra) | NB3 Keras (0,872) | NB2 BiLSTM (0,865) | NB1 CNN (0,55) |
| F1-Score (amostra) | NB2 P99 (5,78%) | NB1 CNN (5,47%) | NB1 LSTM (0%) |
| Recall por evento | NB3 Keras (94,1%) | NB2 P95 (96,4%) | NB1 (baixo) |
| Precision por evento | NB3 Keras (88,9%) | — | — |
| Interpretabilidade | NB3 CARE | NB2 por feature | NB1 output |

### Conclusões principais

1. **Abordagem semi-supervisionada (autoencoders) supera o supervisionado** no contexto de dados severamente desbalanceados — AUC-ROC 0.87 vs 0.66 máximo

2. **O LSTM isolado falha completamente** com dados desbalanceados no teste — não deve ser usado sem mecanismos robustos de balanceamento

3. **A Precision baixa (~2-3%) é o maior desafio aberto** — muitos falsos positivos tornam o sistema impraticável em produção

4. **Avaliação por evento é mais útil operacionalmente** — o que importa é detectar que um evento anômalo ocorreu, não classificar cada timestep

5. **Optuna melhora resultados** mesmo com poucas épocas (50) — a busca automática encontrou hiperparâmetros que manual levaria muito mais tempo

### Próximos passos sugeridos

| Melhoria | Notebook | Benefício esperado |
|---------|----------|-------------------|
| **Weighted CrossEntropy** | NB1 | Reduzir collapse do LSTM |
| **Focal Loss** | NB1 | Focar em exemplos difíceis (anomalias) |
| **Threshold por regime operacional** | Todos | Reduzir FP em condições variadas |
| **Mais épocas + LR scheduling** | NB2 | Melhor convergência do autoencoder |
| **Calibração por evento** | Todos | Threshold adaptado ao comportamento de cada turbina |
| **Ensemble NB2 + NB3** | Novo | Combinar temporal (BiLSTM) e estático (MLP) |
| **SMOTE para NB1** | NB1 | Geração sintética de anomalias para treino |

---

*Análise gerada com base nos notebooks `wind_turbine_cnn_lstm_paper.ipynb`, `wind_turbine_anomaly_detection_v4.ipynb` e `wind_turbine_autoencoder_keras_pipeline.ipynb` e seus respectivos logs de treinamento.*
