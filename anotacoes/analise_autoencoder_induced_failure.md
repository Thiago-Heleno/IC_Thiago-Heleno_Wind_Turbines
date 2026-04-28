# Analise Tecnica - wind_turbine_autoencoder_induced_failure.ipynb

> **Resultados salvos em:** [`resultados/05_autoencoder_induced_failure/`](../resultados/05_autoencoder_induced_failure/)
> **Metricas padronizadas:** [`resultados/05_autoencoder_induced_failure/metricas.json`](../resultados/05_autoencoder_induced_failure/metricas.json)

## Objetivo

Treinar um Autoencoder MLP em regime semi-supervisionado para deteccao de anomalias em SCADA de turbinas eolicas, utilizando falhas sinteticas injetadas na etapa de calibracao para derivar thresholds de deteccao mais robustos.

A hipotese central e que usar o RMSE de reconstrucao de anomalias *induzidas* para calibrar o threshold — em vez de usar apenas o percentil do dado normal — reduz falsos positivos sem sacrificar a deteccao de anomalias reais.

## Dataset

- **Fonte:** CARE_To_Compare / Wind Farm C
- **Arquivos:** CSVs individuais por evento em `datasets/`, acompanhados de `event_info.csv`
- **Labels binarios:** derivados de `event_label` (quando disponivel) ou `status_id in {3,4}`

## Paradigma: Semi-Supervisionado

O autoencoder e treinado **exclusivamente em dados normais**. Nao ha uso de labels durante o treino. As labels sao usadas apenas na avaliacao final e na calibracao do threshold.

Fluxo:

```
Dados normais (treino) --> Autoencoder aprende reconstrucao normal
Dado de entrada (teste)  --> AE reconstroi --> RMSE alto = anomalia
```

## Split dos Dados

O split utiliza a coluna `train_test` do CARE (split temporal pre-definido pelo benchmark):

1. **Treino completo** (`train_test == "train"`)
2. **Teste** (`train_test in ["test", "prediction"]`)

Dentro do conjunto de treino, apenas os dados normais sao usados, e eles sao divididos temporalmente em:

| Conjunto | Fracao | Papel |
|----------|--------|-------|
| Train AE | 80% dos normais de treino | Treinar o autoencoder |
| Calibracao | 20% dos normais de treino | Calibrar o threshold |
| Teste | split CARE original | Avaliacao final |

O split de calibracao e feito temporalmente (nao aleatorio) para preservar a ordem cronologica dos dados SCADA.

## Pre-processamento

Implementado na classe `build_preprocessor()` como Pipeline sklearn:

1. **DataClipper** - Recorta outliers extremos usando quantis [0.001, 0.999] calculados apenas nos dados de treino normal. Estabiliza a escala para o AE.
2. **NanImputer** - Substitui valores faltantes pela media da feature no treino normal.
3. **StandardScaler** - Normaliza cada feature para media zero e desvio padrao unitario.

O preprocessor e fitado **apenas em `X_train_ae`** (dados normais de treino) e aplicado por transformacao nos demais conjuntos.

## Selecao de Features (Nao-supervisionada)

Funcao `select_features_unsupervised()` com duas etapas:

1. **Remocao por variancia baixa** - Features com variancia < 1e-4 sao descartadas.
2. **Remocao por correlacao alta** - Para cada par com correlacao de Pearson > 0.95, remove-se o segundo da correlacao. Calculado em amostra de ate 50.000 pontos.

Feita exclusivamente a partir dos dados normais de treino (nao-supervisionada).

## Arquitetura do Autoencoder

MLP simetrico com estrutura encoder-decoder:

**Encoder:**
- 1 a 3 camadas densas com tamanhos [200, 100, 50] (conforme `n_layers`)
- Ativacao: PReLU em cada camada
- Camada de codigo (bottleneck): `code_size` neuronios com PReLU

**Decoder (espelho do encoder):**
- Camadas densas na ordem inversa do encoder
- Saida linear com `input_dim` neuronios (reconstrucao da entrada)

**Compilacao:**
- Perda: MSE (Mean Squared Error)
- Otimizador: Adam com ExponentialDecay no learning rate
- Metrica auxiliar: MAE

O autoencoder e treinado para reconstruir sua propria entrada. O erro de reconstrucao (RMSE por amostra) e usado como score de anomalia.

## Score de Anomalia

Para cada amostra, o score e o **RMSE de reconstrucao por amostra**:

```
RMSE(x) = sqrt( mean( (x - AE(x))^2 ) )
```

- Amostras normais: o AE as viu durante treino, RMSE baixo.
- Amostras anomalas: o AE nao aprendeu o padrao, RMSE alto.

## Otimizacao de Hiperparametros (Optuna)

Framework: **Optuna** com amostrador **TPE (Tree-structured Parzen Estimator)**, seed 42.

- **Numero de trials:** 30 (`N_OPTUNA_TRIALS`)
- **Metrica:** val_loss (MSE na validacao interna de treino)
- **Direcao:** minimizar

Espaco de busca:

| Hiperparametro | Tipo | Faixa |
|----------------|------|-------|
| `n_layers` | int | 1 a 3 |
| `code_size` | int | 10 a 64 |
| `learning_rate` | float (log) | 1e-4 a 1e-2 |
| `decay_rate` | float | 0.90 a 0.999 |

Em cada trial: split interno 80/20 do train_ae, early stopping patience=5, maximo 30 epochs.

## Treinamento Final do Autoencoder

Com os melhores hiperparametros:

- Ate 200 epochs
- Early stopping patience=10 no val_loss
- Checkpoint do melhor epoch salvo em `autoencoder_best.keras`
- RMSE computado para train, calibration e test apos o treino

## Injecao de Falhas Sinteticas na Calibracao

A injecao e feita **apenas no conjunto de calibracao** (dados normais). Ela nao afeta o treino do AE nem o teste.

### Fracao de Injecao

7% dos dados normais de calibracao recebem perturbacoes (`INJECTION_FRACTION = 0.07`).

### Tipos de Perturbacao

Aplicadas prioritariamente em features com nomes relacionados a grandezas fisicas:

**1. Ruido Gaussiano (`gaussian`)**
- Adiciona ruido com amplitude 3x o desvio padrao da feature.
- Aplicado em blocos contiguos de 12 amostras.

**2. Drift Linear (`drift`)**
- Injeta rampa linear com magnitude de ate 5x o desvio padrao.
- Simula degradacao gradual de componente.

**3. Picos Isolados (`spike`)**
- Picos abruptos com 8x o desvio padrao em pontos isolados.

A distribuicao e uniforme: ~1/3 de cada tipo.

## Calibracao de Threshold — Tres Estrategias

Este e o ponto central do notebook. Tres thresholds sao derivados e comparados:

### 1. Threshold Padrao (P95 Normal)

```
threshold_standard = percentil_95( RMSE_normal_calibracao )
```

Abordagem classica: qualquer amostra com RMSE acima do P95 do comportamento normal e classificada como anomalia.

**Problema:** pode gerar muitos falsos positivos se o dado normal tiver alta variabilidade natural.

### 2. Threshold Induzido (P95 Induzido)

```
threshold_induced = percentil_95( RMSE_anomalias_induzidas )
```

A ideia e: se o modelo nao consegue reconstruir bem nem as falhas *sinteticas* (que sao perturbacoes controladas), o threshold deve estar acima desse nivel para nao alarmar em perturbacoes "comuns". Anomalias reais, por serem mais severas, devem superar esse threshold.

**Vantagem:** menos falsos positivos em perturbacoes triviais. Threshold mais conservador.

### 3. Threshold Adaptativo (P50 Induzido + Gamma)

```
threshold_adaptive = percentil_50( RMSE_induzido ) + gamma
```

O `gamma` e otimizado por uma segunda rodada do Optuna (30 trials) que busca o maior gamma tal que a taxa de falsos positivos nos dados normais de calibracao permaneca abaixo de 5%.

**Vantagem:** controle explicito da taxa de falsos positivos, com threshold adaptado a distribuicao das perturbacoes.

### Comparacao

```
Esperado: RMSE_normal < RMSE_induzido < RMSE_anomalia_real
           threshold_standard < threshold_induced < threshold_adaptive
```

## Avaliacao no Teste

Os tres thresholds sao avaliados no conjunto de teste com as mesmas metricas:

- Precision, Recall, F1-Score, Accuracy
- CARE Score (metrica oficial do benchmark)

### CARE Score

Avalia em nivel de evento (por dataset/CSV):

- **F1_2:** F-beta (beta=0.5) por evento anomalo
- **Acc:** acuracia em dados normais por evento normal
- **EF1_2:** F-beta ao nivel de evento (detectou ou nao o evento)
- **WS (Weighted Score):** pondera o alarme pelo momento no evento (alarme precoce vale mais)
- **CARE:** formula combinada: `(F1_2 + WS + EF1_2 + 2*Acc) / 5`

O threshold com maior CARE e selecionado como melhor e salvo nos artefatos.

## Artefatos Gerados

Salvos em `resultados/05_autoencoder_induced_failure/`:

| Arquivo | Conteudo |
|---------|----------|
| `autoencoder_induced.keras` | Modelo final treinado |
| `autoencoder_best.keras` | Checkpoint do melhor epoch |
| `best_params.json` | Hiperparametros Optuna do AE |
| `threshold_params.json` | Valores dos tres thresholds e gamma |
| `metricas.json` | Metricas completas por threshold |
| `care_results.csv` | CARE por dataset do melhor threshold |
| `care_summary.csv` | CARE agregado do melhor threshold |
| `comparison.csv` | Comparacao dos tres thresholds |
| `comparison_plots.png` | Loss do AE, distribuicao RMSE, serie temporal, matrizes de confusao, CARE sub-scores |

## Relacao com o Notebook 04 (Classifier)

Ambos os notebooks usam as mesmas utilidades (`induced_failure_utils.py`) e o mesmo mecanismo de injecao sintetica, mas com papeis diferentes:

| Aspecto | NB04 Classifier | NB05 Autoencoder |
|---------|----------------|-----------------|
| Paradigma | Supervisionado | Semi-supervisionado |
| Uso das falhas sinteticas | Dado de treino (3a classe) | Calibracao do threshold |
| Score de anomalia | P(classe 1) via softmax | RMSE de reconstrucao |
| Requer labels no treino | Sim (anomalias reais) | Nao (treina so com normais) |
| Threshold | Unico (otimizado na PR curve) | Tres alternativas comparadas |

## Pontos Importantes para a Reuniao

- O autoencoder e treinado em regime **nao-supervisionado** — nao usa nenhuma label de anomalia no treino.
- A inovacao esta na etapa de calibracao: em vez de usar apenas o percentil do dado normal, usamos o RMSE das proprias falhas sinteticas para definir onde o threshold deve estar.
- O threshold adaptativo controla explicitamente a taxa de falsos positivos (FPR < 5%), o que e desejavel em aplicacoes industriais onde alarmes falsos tem custo operacional.
- O Optuna e usado duas vezes: primeiro para otimizar os hiperparametros do AE, depois para encontrar o melhor gamma do threshold adaptativo.
- A abordagem e aplicavel em cenarios onde ha poucos ou nenhum exemplo rotulado de anomalia, pois o treino do AE e completamente nao-supervisionado.
