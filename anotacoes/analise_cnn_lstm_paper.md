# Analise Tecnica - wind_turbine_cnn_lstm_paper.ipynb

## Objetivo

Reproduzir e adaptar a abordagem CNN-LSTM do artigo de Qi et al. (Energies 2024) para deteccao de falhas com dados SCADA do CARE (Wind Farm C), tratando o problema como classificacao binaria:

- Classe 0: normal
- Classe 1: anomalia

## Configuracao Experimental

Parametros principais no notebook:

- Janela temporal: 36 passos (6 horas).
- Batch size: 600.
- Epochs max: 500.
- Learning rate: 1e-3.
- Early stopping patience: 20.
- Selecao de features: top 30% por importancia do XGBoost.

## Dados e Pre-processamento

Arquivos:

- CARE_To_Compare/Wind Farm C/datasets/\*.csv
- event_info.csv
- feature_description.csv

Passos:

1. Carrega metadados de evento e descricao de sensores.
2. Identifica colunas de metadados (META_COLS) e sensores.
3. Trata features angulares com transformacao seno/cosseno.
4. Rotula cada timestamp:
   - label=1 se id no intervalo [event_start_id, event_end_id] para eventos anomalos.
   - caso contrario label=0.
5. Aplica ffill e bfill por arquivo.

## Split e Controle de Leakage

Split feito por evento (event-level):

- 70% eventos treino
- 15% eventos validacao
- 15% eventos teste

Ponto forte:

- O mesmo evento nao aparece em dois splits.
- Isso reduz risco de leakage temporal/estrutural entre treino e teste.

## Normalizacao

Normalizacao Min-Max feita manualmente com estatisticas de treino:

- Computa col_min e col_max apenas em linhas normais dos eventos de treino.
- Aplica para todos os splits.

Importante:

- Evita usar informacao de validacao/teste para ajustar escala.

## Selecao de Features por XGBoost

Estrategia:

1. Amostra ate ~100k linhas dos eventos de treino.
2. Treina XGBClassifier com ajuste de desbalanceamento (scale_pos_weight).
3. Ordena importancias e seleciona top 30%.

Resultado:

- sensor_cols passa a ser apenas subconjunto selecionado.
- Reduz dimensionalidade e custo de treinamento da rede.

## Construcao de Janelas e Definicao de X

Janela:

- Tamanho W=36
- Rotulo de janela: anomalia se qualquer timestamp da janela tiver label=1.

Undersampling:

- Aplicado somente em treino (janelas normais).
- Validacao e teste mantem distribuicao natural.

Representacao de entrada X:

- X shape por amostra = (36, n_features_selecionadas).
- DataLoader monta janelas on-the-fly a partir de arrays flat por evento.

Vantagem:

- Eficiencia de memoria para grande volume de dados.

## Definicao de Y da Rede

No modelo supervisionado CNN-LSTM:

- Y = classe binaria da janela.
- Formato: inteiro 0 ou 1.

Regra de construcao do Y por janela:

- Y=1 se qualquer timestamp dentro da janela de 36 passos for anomalo.
- Y=0 se todos os timestamps da janela forem normais.

No treinamento PyTorch:

- y e passado como tensor do tipo long para CrossEntropyLoss.
- A saida da rede tem 2 logits (classe normal e classe anomalia).

## Arquitetura CNN-LSTM (Option 1)

Modelo principal (CNNLSTM):

1. Conv1d: n_features -> 32, kernel=3.
2. MaxPool1d: 2.
3. Conv1d: 32 -> 64, kernel=2.
4. MaxPool1d: 3.
5. LSTM: 64 -> 100.
6. LSTM: 100 -> 80.
7. Dropout 0.5.
8. Dense: 80 -> 50 -> 10 -> 2.
9. Ativacao SELU nas camadas internas.

Loss e treino:

- CrossEntropyLoss.
- Adam.
- Early stopping por val_loss.

## Ajuste de Threshold e Avaliacao

Saida de probabilidade:

- Softmax, usando probabilidade da classe anomalia.

Calibracao:

- Busca threshold na validacao maximizando F1 via curva precision-recall.

Avaliacao final:

- Aplica threshold otimizado em teste.
- Reporta Accuracy, Precision, Recall, F1, AUC-ROC, matriz de confusao e classification report.

## Baselines para Comparacao

Notebook treina tambem:

- CNNOnly.
- LSTMOnly.

Mesmo esquema geral:

- Mesmos dados de entrada.
- Mesmo split.
- Mesma logica de ajuste de threshold em validacao.

Objetivo:

- Comparar se combinacao CNN+LSTM melhora sobre modelos isolados.

## Artefatos Salvos

Diretorio:

- ../resultados/results_cnn_lstm_paper

Arquivos:

- cnn_lstm_model.pth
- cnn_model.pth
- lstm_model.pth
- scaler.npz
- results.json

## Pontos Fortes

- Split por evento reduz leakage.
- Undersampling restrito ao treino evita inflar metricas de validacao/teste.
- Comparacao com baselines facilita argumento cientifico.

## Pontos de Atencao

- Undersampling agressivo pode reduzir diversidade de janelas normais.
- Threshold ajustado em validacao precisa sempre ser reportado como etapa de calibracao.
- Resultados devem ser interpretados considerando prevalencia real de anomalias no teste.
