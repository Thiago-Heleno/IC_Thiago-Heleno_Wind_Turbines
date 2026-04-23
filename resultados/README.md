# Resultados — Pesquisa de Iniciacao Cientifica

Deteccao e predicao precoce de falhas em turbinas eolicas com dados SCADA do dataset **CARE_To_Compare (Wind Farm C)**.

- **Autor:** Thiago Heleno
- **Orientacao:** IC — 2025/2026
- **Dataset:** [CARE_To_Compare v1.0](https://doi.org/10.5281/zenodo.10958775) — Guck & Roelofs (2024)

---

## Estrutura

```
resultados/
|- README.md                         # este arquivo (indice + comparacao)
|- metricas_consolidadas.csv         # todas as metricas lado a lado
|- 01_cnn_lstm_supervisionado/       # Notebook 1 - supervisionado (CNN / LSTM / CNN-LSTM)
|- 02_cnn_bilstm_autoencoder/        # Notebook 2 - autoencoder CNN-BiLSTM-Attention (PyTorch)
|- 03_keras_mlp_autoencoder/         # Notebook 3 - autoencoder MLP (Keras + Optuna)
```

Cada subpasta contem:

- `metricas.json` — metricas padronizadas (schema uniforme)
- `README.md` — contexto, como reproduzir, principais numeros
- artefatos brutos do experimento (modelos, figuras, thresholds, CSVs)

---

## Schema de `metricas.json`

Todos os notebooks exportam metricas no mesmo formato de topo:

| Campo                 | Descricao                                                   |
|-----------------------|-------------------------------------------------------------|
| `notebook`            | Nome-base do notebook                                       |
| `codigo_fonte`        | Caminho do `.ipynb` que gerou os artefatos                  |
| `paradigma`           | `supervisionado` ou `semi-supervisionado`                   |
| `modelo`              | Arquitetura resumida                                        |
| `dataset`             | Fonte de dados                                              |
| `data_execucao`       | Data (YYYY-MM-DD) em que os artefatos foram produzidos      |
| `configuracao`        | Splits, janelas, pre-processamento                          |
| `hiperparametros`     | HPs finais (Optuna / manual)                                |
| `thresholds`          | Cortes de decisao e parametros de pos-processamento         |
| `metricas_teste`      | Metricas por configuracao / classificador                   |
| `metricas_CARE`       | Metricas no padrao do benchmark CARE (quando aplicavel)     |
| `artefatos`           | Mapa de arquivos gerados na pasta                           |
| `notas`               | Observacoes relevantes para interpretar os numeros          |

---

## Comparacao rapida

> Numeros completos em `metricas_consolidadas.csv`. Metricas abaixo sao do split de **teste**.

| # | Notebook                         | Paradigma            | Modelo               | Metrica principal             | Valor |
|---|----------------------------------|----------------------|----------------------|-------------------------------|-------|
| 1 | `wind_turbine_cnn_lstm_paper`    | Supervisionado       | CNN / LSTM / CNN-LSTM| *pendente (rodar notebook)*   | —     |
| 2 | `wind_turbine_anomaly_detection_v4` | Semi-supervisionado | CNN-BiLSTM-Attention AE | AUC-ROC (teste)            | 0.872 |
| 2 | idem                             | idem                 | idem + ponderacao    | F1 (teste)                    | 0.045 |
| 3 | `wind_turbine_autoencoder_keras_pipeline` | Semi-supervisionado | Dense MLP AE      | Score CARE                    | 0.699 |
| 3 | idem                             | idem                 | idem                 | EF1_2 (event-level)           | 0.899 |

**Leitura dos numeros:**

- F1 baixo no notebook 2 reflete desbalanceamento extremo (~51:1 Normal:Anomalia) e nao falha do modelo — AUC-ROC de 0.87 e informativa.
- Notebook 3 usa metricas do benchmark CARE (F1_2, EF1_2, WS) que ponderam eventos; EF1_2=0.90 indica forte capacidade de deteccao em nivel de evento.
- Notebook 1 (supervisionado) ainda nao foi executado; artefatos serao populados quando o pipeline rodar.

---

## Reproduzindo os resultados

```bash
# 1) ativar venv com requirements do projeto
pip install -r notebooks/requirements.txt

# 2) rodar notebook desejado (exemplos)
jupyter nbconvert --to notebook --execute notebooks/wind_turbine_cnn_lstm_paper.ipynb
jupyter nbconvert --to notebook --execute notebooks/wind_turbine_anomaly_detection_v4.ipynb
jupyter nbconvert --to notebook --execute notebooks/wind_turbine_autoencoder_keras_pipeline.ipynb

# 3) artefatos sao escritos automaticamente em resultados/0X_<nome>/
```

Para analise textual completa de cada pipeline, ver `anotacoes/` na raiz do projeto e `analise_comparativa.md`.
