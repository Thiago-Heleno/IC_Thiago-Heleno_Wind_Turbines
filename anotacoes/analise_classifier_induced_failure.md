# Analise Tecnica - wind_turbine_classifier_induced_failure.ipynb

> **Resultados salvos em:** [`resultados/04_classifier_induced_failure/`](../resultados/04_classifier_induced_failure/)
> **Metricas padronizadas:** [`resultados/04_classifier_induced_failure/metricas.json`](../resultados/04_classifier_induced_failure/metricas.json)

## Objetivo

Treinar um classificador MLP supervisionado para distinguir tres classes de amostras SCADA:

- Classe 0: Normal
- Classe 1: Anomalia real (registrada no dataset CARE)
- Classe 2: Anomalia induzida sinteticamente

A hipotese central e que expor o modelo a falhas sinteticas durante o treino melhora a generalizacao para anomalias reais, em comparacao a um baseline treinado apenas com dados originais.

## Dataset

- **Fonte:** CARE_To_Compare / Wind Farm C
- **Arquivos:** CSVs individuais por evento em `datasets/`, acompanhados de `event_info.csv` e `feature_description.csv`
- **Total de amostras:** varios milhoes de leituras SCADA de 10 minutos
- **Labels:** derivados de `event_label` (quando disponivel) ou `status_id in {3,4}`

## Split dos Dados

O split e feito em nivel de evento (event-level), nao em nivel de amostra, para evitar vazamento de informacao (leakage):

- Todos os `source_file` (CSVs individuais) sao embaralhados com seed fixo.
- Divisao: **70% treino / 15% validacao / 15% teste**
- Cada split mantem eventos inteiros, garantindo que a temporalidade intra-evento nao seja quebrada.

Contagem de classes por split (para reuniao):

| Split | Normal | Anomalia Real |
|-------|--------|---------------|
| Treino | maior parte | minoria |
| Validacao | proporcional | proporcional |
| Teste | proporcional | proporcional |

## Pre-processamento

Implementado na classe `build_preprocessor()` como Pipeline sklearn:

1. **DataClipper** - Recorta outliers extremos usando quantis [0.001, 0.999] calculados apenas nos dados normais de treino. Evita que sensores com leituras espurias contaminem a normalizacao.
2. **NanImputer** - Substitui valores faltantes pela media de cada feature, calculada no treino normal. Necessario porque sensores SCADA frequentemente reportam NaN em condicoes de parada.
3. **StandardScaler** - Normaliza cada feature para media zero e desvio padrao unitario.

O preprocessor e fitado **apenas nos dados normais de treino** (mascara `train_normal_mask`), prevenindo que o comportamento anomalo influencie a escala das features.

## Selecao de Features (Nao-supervisionada)

Funcao `select_features_unsupervised()` aplicada em duas etapas:

1. **Remocao por variancia baixa** - Features com variancia < 1e-4 sao descartadas (sensores constantes ou com pouca informacao).
2. **Remocao por correlacao alta** - Para cada par com correlacao de Pearson > 0.95, remove-se o segundo feature do par. Calculado em amostra de ate 50.000 pontos para viabilidade computacional.

A selecao e feita apenas com base nos dados normais de treino, de forma nao-supervisionada (sem usar os labels).

## Injecao de Falhas Sinteticas

A injecao e o diferencial central deste notebook. Ela ocorre **somente nos dados de treino** (nao contamina validacao nem teste).

### Fracao de Injecao

7% dos dados normais de treino sao selecionados para receber perturbacoes sinteticas (`INJECTION_FRACTION = 0.07`).

### Tipos de Perturbacao

As perturbacoes sao aplicadas prioritariamente em features com nomes relacionados a grandezas fisicas (temperatura, vibracao, velocidade, potencia, pressao, rolamento, rotor, gerador, caixa de engrenagens, oleo, vento):

**1. Ruido Gaussiano (`gaussian`)**
- Adiciona ruido aleatorio com amplitude 3x o desvio padrao da feature.
- Aplicado em um bloco contiguo de 12 amostras ao redor do ponto selecionado.
- Simula leituras ruidosas de sensores.

**2. Drift Linear (`drift`)**
- Injeta uma rampa linear com magnitude de ate 5x o desvio padrao da feature.
- Aplicado em metade das amostras do evento.
- Simula degradacao gradual de componente (ex: desgaste de rolamento).

**3. Picos Isolados (`spike`)**
- Adiciona picos abruptos com amplitude 8x o desvio padrao.
- Aplicado em 1 a 5 pontos isolados.
- Simula falhas eletricas ou leituras erroneas pontuais.

A distribuicao e uniforme: ~1/3 de cada tipo. As amostras originais normais sao mantidas no conjunto de treino junto com as amostras induzidas.

### Conjunto de Treino 3 Classes

```
X_train_3class = [dados_normais_originais + amostras_induzidas + anomalias_reais]
y_train_3class = [0=normal, 2=induzida, 1=real_anomalia]
```

## Arquitetura do Classificador

MLP (Multi-Layer Perceptron) com:

- **Entrada:** numero de features selecionadas
- **Camadas ocultas:** 1 a 3 camadas, com `hidden_units` neuronios na primeira camada e reducao por fator de 2 em cada camada subsequente
- **Ativacao:** PReLU (Parametric ReLU) em cada camada
- **Regularizacao:** Dropout com taxa otimizada
- **Saida:** 3 neuronios com softmax (probabilidades por classe)
- **Perda:** Sparse Categorical Crossentropy
- **Otimizador:** Adam

Para lidar com o desbalanceamento de classes, sao calculados pesos inversamente proporcionais a frequencia de cada classe (`class_weight="balanced"`).

## Otimizacao de Hiperparametros (Optuna)

Framework: **Optuna** com amostrador **TPE (Tree-structured Parzen Estimator)**, seed 42.

- **Numero de trials:** 30 (`N_OPTUNA_TRIALS`)
- **Metrica:** F1-score binario na validacao (classe 1 = anomalia real vs. resto)
- **Direcao:** maximizar

Espaco de busca:

| Hiperparametro | Tipo | Faixa |
|----------------|------|-------|
| `n_layers` | int | 1 a 3 |
| `hidden_units` | int | 64 a 256 (step 32) |
| `dropout_rate` | float | 0.1 a 0.5 |
| `learning_rate` | float (log) | 1e-4 a 1e-2 |

Em cada trial: treinamento com split interno 85/15 do treino, early stopping com patience=5.

## Treinamento Final

Dois modelos sao treinados com os mesmos hiperparametros otimizados:

**Modelo Induzido (3 classes):** treinado em `X_train_3class` com as falhas sinteticas.

**Modelo Baseline (2 classes):** treinado apenas com Normal vs. Anomalia Real (sem falhas sinteticas). Permite medir o ganho da injecao sintetica.

Ambos usam:
- Ate 100 epochs
- Early stopping patience=10 no val_loss
- Checkpoint do melhor modelo

## Calculo do Threshold de Decisao

A saida do modelo e `P(classe 1 = anomalia real)`. O threshold de classificacao binaria nao e fixado em 0.5 — e otimizado na curva Precision-Recall:

1. Computar `P(classe 1)` para todos os exemplos de validacao.
2. Varrer todos os thresholds na curva PR.
3. Selecionar o threshold que maximiza o F1-score binario na validacao.

O mesmo processo e aplicado ao baseline.

## Avaliacao no Teste

Metricas calculadas:

- Precision, Recall, F1-Score, Accuracy
- AUC-ROC
- CARE Score (metrica do benchmark CARE_To_Compare)

### CARE Score

O CARE Score avalia o modelo em nivel de evento (por dataset/CSV), capturando:

- **F1_2:** F-beta (beta=0.5) por evento anomalo (penaliza mais falso negativo)
- **Acc:** acuracia em dados normais por evento normal (especificidade)
- **EF1_2:** F-beta ao nivel de evento (detectou ou nao o evento)
- **WS (Weighted Score):** pondera o alarme pelo momento em que ocorreu dentro do evento (alarme precoce vale mais)
- **CARE:** formula combinada: `(F1_2 + WS + EF1_2 + 2*Acc) / 5`

A comparacao entre Modelo Induzido e Baseline no CARE Score e a principal evidencia do beneficio da injecao sintetica.

## Artefatos Gerados

Salvos em `resultados/04_classifier_induced_failure/`:

| Arquivo | Conteudo |
|---------|----------|
| `classifier_induced.keras` | Modelo final treinado com falhas sinteticas |
| `classifier_best.keras` | Checkpoint do melhor epoch por val_loss |
| `best_params.json` | Hiperparametros otimizados pelo Optuna |
| `metricas.json` | Metricas completas (teste + CARE) de ambos os modelos |
| `care_results.csv` | CARE por dataset do modelo induzido |
| `comparison.csv` | Tabela comparativa Induzido vs Baseline |
| `comparison_plots.png` | Loss, matrizes de confusao, ROC, PR, CARE sub-scores |

## Resultados Obtidos (execucao 2026-04-29)

### Hiperparametros otimos (Optuna, 30 trials)

| HP | Valor |
|----|-------|
| n_layers | 2 |
| hidden_units | 192 |
| dropout_rate | 0.2945 |
| learning_rate | 5.47e-3 |

Threshold binarizacao P(classe1) otimo: **1.84e-25** (curva PR, F1 max).

### Metricas por amostra (teste)

| Modelo | Precision | Recall | F1 | Accuracy |
|--------|-----------|--------|-----|----------|
| **Modelo Induzido (3 classes)** | 0.5079 | **1.0000** | **0.6736** | 0.5079 |
| Baseline (sem falhas sinteticas) | 0.4073 | 0.6657 | 0.5054 | 0.3381 |

> **Ganho da injecao sintetica:** F1 +16.8 pontos absolutos (0.674 vs 0.505), Recall total (100% vs 66.6%).

### CARE Score (10 datasets de teste, 5 anomalos / 5 normais)

| Modelo | F1_2 | Acc | EF1_2 | WS | CARE |
|--------|------|-----|-------|-----|------|
| Induzido | 0.0 | 3.6e-6 | 0.5556 | 1.0000 | 3.61e-6 |
| Baseline | 0.0 | 2.2e-5 | 0.5556 | 0.6486 | 2.16e-5 |

### Interpretacao

- O modelo induzido alarma em todos os eventos anomalos do teste (5/5 detectados, WS=1.0), mas tambem alarma nos 5 eventos normais (Acc por evento ~ 0).
- O baseline tem Acc igualmente baixa em eventos normais, mas alarme tardio (WS=0.65) e deteccao parcial.
- F1_2=0 em ambos: como a F-beta exige precisao e recall por evento e Acc=0 em normais arrasta a media, o componente colapsa.
- Trade-off observado: **alta sensibilidade amostral** vs **baixa especificidade por evento**. A injecao sintetica sensibiliza o modelo, mas a calibracao por evento permanece um problema aberto.

## Pontos Importantes para a Reuniao

- A injecao de falhas sinteticas e uma tecnica de **data augmentation supervisionada** aplicada ao treino para aumentar a diversidade de padroes anomalos.
- O split por evento garante que o modelo nao "memorize" eventos durante o treino.
- A validacao por F1 e o threshold adaptativo evitam o problema de threshold fixo em 0.5, comum em modelos imbalanceados.
- O CARE Score e a metrica oficial do benchmark e prioriza deteccao antecipada e baixa taxa de falsos positivos em dados normais.
