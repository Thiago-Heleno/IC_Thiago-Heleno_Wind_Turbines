# Analise Tecnica - wind_turbine_autoencoder_keras_pipeline.ipynb

## Objetivo

Construir um pipeline end-to-end de deteccao de anomalias em SCADA com:

- Autoencoder MLP (Keras/TensorFlow).
- Threshold fixo e adaptativo.
- Explicabilidade local com ARCANA.
- Avaliacao agregada via CARE Score por sub-dataset.

## Fontes de Dados e Ingestao

Base:

- CARE_To_Compare/Wind Farm C/datasets
- CARE_To_Compare/Wind Farm C/event_info.csv

Boas praticas implementadas:

- Deteccao automatica de delimitador (; ou ,).
- Normalizacao de schema CARE:
  - time_stamp -> timestamp
  - status_type_id -> status_id
- Downcast de tipos numericos para reduzir memoria.
- Inclusao de source_file para rastrear qual CSV gerou cada amostra.

## Split Temporal do Pipeline

1. Separa train/test com base em coluna train_test.
2. Filtra train normal para treinar o AE (remove status anomalo 3/4 nessa etapa).
3. Divide train normal temporalmente em:
   - Train AE (80%)
   - Calibration (20%)

Papel de cada conjunto:

- Train AE: aprender padrao de reconstrucao normal.
- Calibration: calibrar threshold fixo/adaptativo.
- Test: avaliacao final.

## Selecao de Colunas e Labels

FEATURE_COLS:

- Inferidas automaticamente como colunas numericas nao-meta.

infer_binary_labels:

- Prioriza event_label (quando existe).
- Fallback: status_id em {3,4} como anomalia.

## Etapa de Pre-processamento

### 2a) DataClipper (antes do Pipeline sklearn)

Clipping por quantis no treino:

- lower_q=0.001
- upper_q=0.999
- asset_id preservado

Objetivo:

- Reduzir impacto de outliers extremos sem destruir estrutura temporal.

### 2b) DataPreprocessor (Pipeline scikit-learn)

Ordem aplicada no notebook:

1. DuplicateValuesToNan:
   - Sequencias longas de valor repetido (default 0.0) viram NaN.
2. ColumnSelector:
   - Remove colunas com excesso de NaN.
3. CounterDiffTransformer:
   - Derivada em colunas de contadores (trata reset).
4. RollingFeaturesTransformer:
   - Media/std movel (janela 6) em colunas-chave.
5. LowUniqueValueFilter:
   - Remove colunas quase constantes ou dominadas por zero.
6. AngleTransformer:
   - Converte angulos em sin/cos.
7. TimeSeriesImputer:
   - Interpolacao temporal + fallback por media.
8. DropColumnsTransformer:
   - Remove asset_id.
9. StandardScaler:
   - Padronizacao final (media 0, desvio 1).

Saidas processadas:

- X_train_ae_proc
- X_cal_proc
- X_test_proc

## Definicao de X da Rede

Neste notebook, X e tabular por timestamp (nao em janelas fixas):

- X shape = (n_amostras, n_features_processadas)

Cada linha representa um instante SCADA apos engenharia de atributos e padronizacao.

## Definicao de Y da Rede

No treino do autoencoder:

- Y_treino = X_treino.
- A tarefa e reconstruir a propria entrada tabular.

Na calibracao e teste de deteccao:

- Y_cal e Y_test = labels binarias (0 normal, 1 anomalia).
- Fonte do Y:
  - event_label quando disponivel.
  - fallback por status_id em {3,4} para classe anomala.

No threshold adaptativo:

- Existe tambem um alvo auxiliar de regressao:
  - Y_regressao = RMSE observado em dados normais da calibracao.
  - Esse alvo treina a RegressionNN que estima rmse_pred.

## Autoencoder MLP (Keras)

Construcao:

- Encoder denso com PReLU (camadas dependem de n_layers).
- Bottleneck com dimensao code_size.
- Decoder simetrico.
- Saida linear com dimensao original de entrada.

Treinamento:

- Loss: MSE.
- Otimizador: Adam com ExponentialDecay no learning rate.
- EarlyStopping em val_loss.
- Shuffle=False para respeitar ordem temporal.

## Otimizacao com Optuna

Busca de hiperparametros do AE:

- n_layers
- code_size
- learning_rate
- decay_rate
- gamma

Funcao objetivo do trial:

1. Treina AE em parte temporal de X_train_ae_proc.
2. Reconstrui calibracao e calcula rmse_cal por amostra.
3. Treina rede de regressao para prever RMSE esperado em condicao normal.
4. Predicao anomala quando rmse_cal > (rmse_pred_cal + gamma).
5. Maximiza F-beta (beta=0.5) na calibracao.

## Thresholds de Deteccao

### Threshold fixo

Dois calculos no notebook:

- fixed_threshold_p95: percentil 95 do RMSE em condicao normal.
- fixed_threshold_fbeta: threshold que maximiza F-beta na calibracao.

### Threshold adaptativo

Passos:

1. Treina RegressionNN para estimar RMSE esperado (rmse_pred) dado X.
2. Define limite adaptativo por amostra:
   - limite_adaptativo = rmse_pred + gamma
3. gamma e otimizado por Optuna na calibracao.

Regra final adaptativa:

- pred=1 se rmse > (rmse_pred + gamma)

## ARCANA (Explicabilidade)

Objetivo:

- Identificar quais features mais contribuem para anomalia em amostras detectadas.

Mecanismo:

- Congela AE.
- Otimiza um bias b no input para minimizar:
  - termo de reconstrucao + penalizacao do proprio bias.
- Importancia por feature = |b_i|.

Saida:

- Ranking top-N features por amostra anomala.

## CARE Score por Sub-dataset

Avaliacao agrupada por source_file (cada CSV/evento):

- F1/2 por dataset anomalo.
- Accuracy em partes normais.
- WS (weighted score de earliness) para detectar quao cedo houve alarme no evento.
- EF1/2 no nivel evento (evento com alarme vs sem alarme).

Formula implementada:

- CARE = (F1_2 + WS + EF1_2 + 2\*Acc) / 5
- Com regras de seguranca para caso sem alarme ou acc baixa.

## Artefatos Salvos

Diretorio:

- resultados/results_keras_pipeline

Principais arquivos:

- preprocessor_pipeline.pkl
- data_clipper.pkl
- autoencoder.h5
- threshold_params.json
- care_results.csv
- arcana_results.csv
- best_params.json
- care_summary.json

## Pontos Fortes

- Pipeline completo, com preprocessamento robusto para dados reais.
- Threshold adaptativo reduz dependencia de limiar unico global.
- ARCANA adiciona interpretabilidade.
- CARE Score aproxima a avaliacao de criterio operacional por evento.

## Pontos de Atencao

- Nao usar calibracao como resultado final; ela serve para ajuste de threshold.
- O desempenho pode variar com qualidade de labels de evento/status.
- Threshold adaptativo depende da qualidade da RegressionNN.
