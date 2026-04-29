# Predicao de Falhas em Turbinas Eolicas com SCADA

Este repositorio concentra os experimentos da pesquisa de deteccao e predicao precoce de falhas em turbinas eolicas usando dados SCADA do dataset CARE_To_Compare.

O foco esta em dois paradigmas complementares:

- Classificacao supervisionada (CNN, LSTM e CNN-LSTM).
- Deteccao semi-supervisionada de anomalias (autoencoders treinados em dados saudaveis).

## Dataset: CARE_To_Compare

Referencia:
Guck, C., and Roelofs, C. (2024). Wind Turbine SCADA Data For Early Fault Detection (v1.0). Zenodo. https://doi.org/10.5281/zenodo.10958775

Resumo pratico usado nos notebooks:

- 95 datasets (eventos), com 44 eventos anomalos e 51 normais.
- Frequencia de 10 minutos por amostra.
- Tres parques eolicos (A, B e C).
- Para esta pesquisa, os notebooks principais usam Wind Farm C (58 eventos).
- Cada CSV contem dados de uma turbina em um evento com colunas de sensores, metadados e indicadores operacionais.

Informacoes importantes do README do CARE incorporadas ao pipeline:

- Delimitador padrao dos CSVs: ponto e virgula (;).
- Rotulo de evento (normal/anomaly) vem de event_info.csv.
- Eventos possuem intervalo temporal com inicio/fim e IDs de inicio/fim.

## Estrutura do Projeto

- `notebooks/`: notebooks principais dos experimentos (`.ipynb` + `.py` + log `.txt`).
- `anotacoes/`: analises tecnicas em Markdown para apresentacao academica.
- `CARE_To_Compare/`: dataset.
- `resultados/`: artefatos padronizados por notebook. Ver [resultados/README.md](resultados/README.md).

**Mapa notebook → pasta de resultados:**

| Notebook | Pasta de resultados | Paradigma |
|----------|---------------------|-----------|
| `wind_turbine_cnn_lstm_paper.ipynb` | `resultados/01_cnn_lstm_supervisionado/` | Supervisionado |
| `wind_turbine_anomaly_detection_v4.ipynb` | `resultados/02_cnn_bilstm_autoencoder/` | Semi-supervisionado |
| `wind_turbine_autoencoder_keras_pipeline.ipynb` | `resultados/03_keras_mlp_autoencoder/` | Semi-supervisionado |
| `wind_turbine_classifier_induced_failure.ipynb` | `resultados/04_classifier_induced_failure/` | Supervisionado + Falha Induzida |
| `wind_turbine_autoencoder_induced_failure.ipynb` | `resultados/05_autoencoder_induced_failure/` | Semi-supervisionado + Falha Induzida |

Cada subpasta contem `metricas.json` (schema uniforme) e `README.md` proprio. Comparacao agregada em [resultados/metricas_consolidadas.csv](resultados/metricas_consolidadas.csv).

## Notebooks Principais e Objetivo

### notebooks/wind_turbine_anomaly_detection_v4.ipynb

Pipeline semi-supervisionado com autoencoder CNN-BiLSTM-Attention (PyTorch). Evolucao da v3 com feature-weighted score, threshold Beta-F1, suavizacao temporal e XGBoost pos-hoc.

Decisoes-chave:

- Split temporal global 70/15/15.
- Scaler MinMax ajustado apenas em amostras saudaveis de treino.
- Selecao de features nao-supervisionada em 3 etapas (variancia, correlacao com variaveis operacionais, remocao de redundancia).
- Janela deslizante de 36 passos (6 horas).
- Deteccao por erro de reconstrucao por feature (P95/P99) e score continuo max(error_i/threshold_i).

### notebooks/wind_turbine_autoencoder_keras_pipeline.ipynb

Pipeline com Keras/TensorFlow, DataPreprocessor em Pipeline scikit-learn, threshold adaptativo e ARCANA.

Decisoes-chave:

- Ingestao robusta + normalizacao de schema CARE (timestamp/status_id).
- Split temporal em treino de AE e calibracao.
- Preprocessamento com clipping, imputacao temporal, transformacao angular, derivada de contadores, rolling features e padronizacao.
- Autoencoder MLP simetrico otimizado com Optuna.
- Threshold fixo e adaptativo (rmse_pred + gamma) com regressao auxiliar.
- CARE Score agregado por sub-dataset.

### notebooks/wind_turbine_cnn_lstm_paper.ipynb

Reproducao da estrategia do artigo Qi et al. (Energies 2024) para classificacao supervisionada.

Decisoes-chave:

- Split por evento (event-level) 70/15/15 para evitar leakage.
- Normalizacao com estatisticas de treino normal.
- Selecao de features por XGBoost (top 30%).
- Janela de 36 passos e undersampling apenas no treino.
- Comparacao CNN vs LSTM vs CNN-LSTM com ajuste de threshold na validacao.

## Pipeline da Pesquisa (visao integrada)

1. Ingestao dos CSVs por evento e unificacao de metadados.
2. Definicao do tipo de split (temporal global ou por evento, dependendo do experimento).
3. Pre-processamento sem vazamento (fit apenas no treino apropriado).
4. Definicao de X e Y da rede:
   - Autoencoder v3: X = janela [36, n_features_selecionadas].
     Y (treino do AE) = o proprio X da janela (reconstrucao).
     Y (avaliacao de deteccao) = label binario da janela (0 normal, 1 anomalia).
   - Keras pipeline: X = vetor tabular pre-processado por timestamp.
     Y (treino do AE) = o proprio X tabular (reconstrucao).
     Y (calibracao/avaliacao) = label binario (event_label ou status_id mapeado para 0/1).
   - CNN-LSTM paper: X = janela [36, n_features_top30].
     Y = classe da janela (0 normal, 1 anomalia).
5. Treinamento do modelo (AE ou classificador) com seeds fixas.
6. Calibracao de threshold em validacao (quando aplicavel).
7. Avaliacao final em teste com metricas e analise por evento/dataset.
8. Exportacao de artefatos para reproducibilidade.

## Principios Metodologicos

- Evitar leakage:
  - Split definido antes do fit de scaler/seletores.
  - Val/Test sem undersampling artificial na avaliacao final.
- Consistencia temporal:
  - Janelas respeitam ordenacao cronologica por evento.
- Robustez numerica:
  - Tratamento de NaN/Inf, clipping e imputacao.
- Reproducibilidade:
  - Seeds fixas e configuracao deterministica quando possivel.

## Dependencias

Arquivo de dependencias em notebooks/requirements.txt.

## Documentacao Tecnica Complementar

- [anotacoes/analise_cnn_lstm_paper.md](anotacoes/analise_cnn_lstm_paper.md) — supervisionado CNN/LSTM/CNN-LSTM
- [anotacoes/analise_cnn_bilstm_autoencoder_v4.md](anotacoes/analise_cnn_bilstm_autoencoder_v4.md) — autoencoder CNN-BiLSTM-Attention v4
- [anotacoes/analise_autoencoder_keras.md](anotacoes/analise_autoencoder_keras.md) — autoencoder Keras + Optuna + CARE Score
- [anotacoes/analise_classifier_induced_failure.md](anotacoes/analise_classifier_induced_failure.md) — classifier MLP 3-class com falhas induzidas
- [anotacoes/analise_autoencoder_induced_failure.md](anotacoes/analise_autoencoder_induced_failure.md) — autoencoder MLP com calibracao por falhas induzidas
- [analise_comparativa.md](analise_comparativa.md) — comparacao integrada dos pipelines
- [resultados/README.md](resultados/README.md) — indice de resultados e metricas padronizadas
