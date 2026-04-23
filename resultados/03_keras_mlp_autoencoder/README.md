# 03 — Keras MLP Autoencoder + Threshold Adaptativo

Pipeline **semi-supervisionado** com autoencoder denso simetrico, otimizacao via Optuna, threshold adaptativo por rede auxiliar e avaliacao no padrao do benchmark **CARE**.

- **Notebook:** [notebooks/wind_turbine_autoencoder_keras_pipeline.ipynb](../../notebooks/wind_turbine_autoencoder_keras_pipeline.ipynb)
- **Script:** [notebooks/wind_turbine_autoencoder_keras_pipeline.py](../../notebooks/wind_turbine_autoencoder_keras_pipeline.py)
- **Data de execucao:** 2026-04-09

## Numeros principais (58 datasets)

| Metrica | Valor  | Significado                                          |
|---------|--------|------------------------------------------------------|
| F1_2    | 0.393  | F-beta com beta=2 (prioriza recall)                  |
| Acc     | 0.774  | Acuracia no nivel de dataset                         |
| EF1_2   | **0.899** | F1_2 agregada por evento (event-level)            |
| WS      | 0.657  | Weighted score do benchmark CARE                     |
| CARE    | **0.699** | Score CARE agregado (metrica final)               |

**Datasets:** 58 total — 51 anomalos + 7 normais.

## Hiperparametros otimos (Optuna)

```json
{
  "n_layers": 2,
  "code_size": 35,
  "learning_rate": 9.80e-4,
  "decay_rate": 0.994,
  "gamma": 0.076
}
```

## Threshold adaptativo

Formula: `score(t) = rmse_pred(t) + gamma`, com `gamma = 0.4997` e regressao auxiliar (1 camada, 32 unidades, Adam). Score padronizado (mu=0.279, sigma=0.102). Janela de suavizacao K=5. Criticidade minima=6.

## Arquivos

| Arquivo                      | Descricao                                                  |
|------------------------------|------------------------------------------------------------|
| `metricas.json`              | Metricas no schema padrao do projeto                       |
| `care_summary.json`          | Resumo das metricas CARE (original)                        |
| `care_results.csv`           | CARE por dataset (58 linhas)                               |
| `arcana_results.csv`         | Resultados ARCANA (interpretabilidade)                     |
| `best_params.json`           | Hiperparametros vencedores do Optuna                       |
| `threshold_params.json`      | Parametros do threshold adaptativo                         |
| `optuna_ae_study.db`         | Historico completo do estudo Optuna                        |
| `autoencoder.h5`             | Modelo treinado (HDF5 legacy)                              |
| `autoencoder_best.keras`     | Modelo treinado (formato Keras 3)                          |
| `analysis.md`                | Analise textual detalhada                                  |
