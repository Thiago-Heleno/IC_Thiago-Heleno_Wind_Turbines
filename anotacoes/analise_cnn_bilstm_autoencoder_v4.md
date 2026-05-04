# Analise Tecnica - wind_turbine_anomaly_detection_v4.ipynb

> **Resultados salvos em:** [`resultados/02_cnn_bilstm_autoencoder/`](../resultados/02_cnn_bilstm_autoencoder/)
> **Metricas padronizadas:** [`resultados/02_cnn_bilstm_autoencoder/metricas.json`](../resultados/02_cnn_bilstm_autoencoder/metricas.json)

## Objetivo

Implementar deteccao de anomalias semi-supervisionada em dados SCADA da Wind Farm C usando autoencoder temporal.

Ideia central:

- Treinar o modelo apenas com janelas saudaveis.
- Usar erro de reconstrucao para detectar desvios do comportamento normal.

## Dados e Rotulagem

Base utilizada:

- CARE_To_Compare/Wind Farm C/datasets/\*.csv
- CARE_To_Compare/Wind Farm C/event_info.csv

Processo de rotulagem por timestamp:

1. Para cada CSV (event_id), inicializa label=0.
2. Se o evento for anomalo, marca label=1 no intervalo [event_start, event_end].
3. Define is_healthy=1 quando label=0 e status_type_id em {0, 2}.

Interpretacao:

- label indica anomalia supervisionada (usada para avaliacao).
- is_healthy define dados permitidos para treino do autoencoder.

## Split e Controle de Leakage

Split temporal global por timestamp:

- Treino: 70%
- Validacao: 15%
- Teste: 15%

Detalhes metodologicos:

- Os cortes sao calculados na serie temporal global concatenada.
- Escalonamento e selecao de features sao ajustados no treino (parte saudavel).
- Metricas finais devem ser as de teste.

Observacao importante para apresentacao:

- O notebook verifica explicitamente eventos que cruzam fronteiras de split.
- Isso pode dividir um mesmo evento entre splits, mas sem leakage direto de preprocessamento se o fit continuar restrito ao treino.

## Pre-processamento

### 1) Tratamento de faltantes

Em cada arquivo:

- Forward fill.
- Backward fill.

### 2) Normalizacao

Metodo:

- MinMaxScaler com feature_range=(0,1).

Ajuste:

- Fit somente em df_train com is_healthy=1.

Aplicacao:

- Transform em chunks para treino, validacao e teste.
- Sem clipping em val/test (valores fora de [0,1] sao mantidos).

Justificativa:

- Preserva possivel sinal anomalo fora do range observado em operacao normal de treino.

## Selecao de Features (Nao-supervisionada)

A selecao e feita somente no treino saudavel.

Etapas:

1. Filtro de variancia:
   - Remove features com variancia < 1e-4.
2. Correlacao com variaveis operacionais:
   - Mantem features com correlacao minima (|r| >= 0.1) com variaveis operacionais (ex.: wind speed, power, rotor speed).
3. Remocao de redundancia:
   - Para pares com correlacao alta (|r| > 0.95), remove a menos relevante operacionalmente.

Resultado:

- selected_features (vetor final usado no modelo).
- Reducao de dimensionalidade para melhorar treino, custo computacional e sinal/ruido.

## Definicao de X da Rede

Depois da selecao:

- Cada split contem colunas:
  - Metadados: time_stamp, event_id, label, is_healthy.
  - Features: selected_features.

Entrada para rede:

- Janela deslizante de tamanho WINDOW_SIZE=36.
- Como cada passo e de 10 min, uma janela representa 6 horas.

Formato de X por batch:

- X shape = (batch_size, 36, n_features).

Rotulos de janela (para avaliacao):

- Janela anomala se qualquer timestamp da janela tiver label=1.
- Janela saudavel se todos os 36 passos tiverem is_healthy=1.

Observacao de engenharia:

- O notebook nao materializa todo tensor de janelas na RAM.
- Usa indice compacto (event_id, start) + arrays por evento para montar janela on-the-fly.

## Definicao de Y da Rede

No treino do autoencoder:

- Y_treino = X_treino.
- Ou seja, a rede aprende reconstrucao da propria entrada (aprendizado semi-supervisionado de padrao normal).

Na avaliacao de deteccao:

- Y_avaliacao = rotulo binario da janela (0 normal, 1 anomalia).
- Esse Y vem da regra de janela baseada em labels por timestamp:
  - 1 se qualquer passo da janela for anomalo.
  - 0 caso contrario.

## Arquitetura da Rede

Modelo: CNNBiLSTMAttentionAutoencoder (PyTorch)

Encoder:

- Conv1d: n_features -> 64, kernel=3, padding=1.
- Conv1d: 64 -> 128, kernel=3, padding=1.
- BiLSTM: 128 -> 256 (saida bidirecional).
- BiLSTM: 256 -> 128.

Atencao:

- MultiHeadAttention (embed_dim=128, num_heads=4).
- Residual + LayerNorm.

Decoder:

- LSTM: 128 -> 128.
- LSTM: 128 -> 128.
- FC: 128 -> 256 -> n_features.

Ativacao:

- SELU.

Inicializacao:

- Kaiming normal para camadas lineares/convolucionais.

Ponto importante:

- A sequencia temporal completa e preservada no encoder/decoder (sem mean pooling final).

## Treinamento

Configuracao principal:

- Loss: MAE (L1Loss).
- Otimizador: Adam, lr=1e-4.
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=3).
- Epochs maximas: 100.
- Early stopping: patience=15.
- Batch size: 64.
- Gradient clipping: max_norm=1.0.

Dados de treino:

- Apenas janelas saudaveis (is_healthy=1).

## Deteccao de Anomalia

Erro por janela:

- Reconstrucao por feature: MAE medio no tempo.
- pf_errors shape = (n_windows, n_features).

Thresholds:

- Per-feature P95 e P99 calculados em janelas saudaveis de validacao.

Regras de decisao:

1. Regra per-feature:
   - Conta quantas features excedem seu threshold.
   - Classifica anomalia se contagem >= MIN_FEATURES_EXCEED.
2. Regra por score continuo:
   - score = max(error_i / threshold_i).
   - Threshold global otimizado por curva Precision-Recall (Best-F1 na validacao).

## Metricas

No teste e validacao:

- Accuracy, Precision, Recall, F1.
- AUC-ROC e AUC-PR para score continuo.
- Matriz de confusao.
- Taxa de deteccao por evento (event-level).

Mensagem metodologica para defesa:

- Threshold Best-F1 e calibrado na validacao.
- Portanto, validacao pode estar otimista.
- O resultado final reportavel e o desempenho em teste.

## Artefatos Salvos

Diretorio de saida:

- ../resultados/02_cnn_bilstm_autoencoder

Arquivos principais:

- modelo_cnn_bilstm_attention_autoencoder.pth
- scaler.pkl
- features_selecionadas.json
- thresholds_per_feature_p95.npy
- thresholds_per_feature_p99.npy
- thresholds.json

## Pontos Fortes do Pipeline

- Semi-supervisionado: util quando falhas rotuladas sao escassas.
- Controle de leakage no fit de scaler/selecao.
- Deteccao explicavel por feature (quais sensores excederam threshold).
- Analise por evento, util para manutencao preditiva.

## Limites e Cuidados

- Split temporal global pode dividir um mesmo evento em multiplos splits.
- Dependencia da calibracao de threshold em validacao.
- Algumas features podem reconstruir quase constante (flat), exigindo diagnostico de capacidade do decoder e qualidade de features.
