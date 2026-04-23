# 02 — CNN-BiLSTM-Attention Autoencoder (v4)

Pipeline **semi-supervisionado** com autoencoder profundo treinado apenas em janelas saudaveis. Deteccao por erro de reconstrucao ponderado + XGBoost pos-hoc.

- **Notebook:** [notebooks/wind_turbine_anomaly_detection_v4.ipynb](../../notebooks/wind_turbine_anomaly_detection_v4.ipynb)
- **Script:** [notebooks/wind_turbine_anomaly_detection_v4.py](../../notebooks/wind_turbine_anomaly_detection_v4.py)
- **Data de execucao:** 2026-04-09
- **FAST_MODE:** false (resultados representativos)

## Melhorias da v4 sobre v3

1. Feature-weighted anomaly score (pesos por AUROC individual)
2. Threshold Beta-F1 (beta=0.5) em vez de P95/P99 fixos
3. Suavizacao temporal (K=5)
4. Mascara de features informativas (std > 0.01)
5. Classificador XGBoost pos-hoc sobre o score

## Numeros principais (teste)

| Configuracao              | Precision | Recall  | F1      | AUC-ROC |
|---------------------------|-----------|---------|---------|---------|
| v3 P95 baseline           | 0.0164    | 0.9646  | 0.0322  | 0.8643  |
| v3 P99 baseline           | 0.0299    | 0.8846  | 0.0578  | 0.8643  |
| **v4 Score Pond. + BF1**  | **0.0232**| **0.8826** | **0.0452** | **0.8717** |
| v4 + Suavizacao K=5       | 0.0232    | 0.8826  | 0.0452  | —       |
| v4 XGBoost                | 0.0       | 0.0     | 0.0     | 0.7429  |

**Melhor configuracao:** `v4_ScorePond_BF1` — maior F1 entre variantes v4 com AUROC superior ao baseline.

## Interpretacao

- Desbalanceamento ~51:1 domina a metrica F1; AUC-ROC e mais informativa.
- Recall alto + precision baixa: modelo detecta quase todas as anomalias mas tambem marca muito ruido normal.
- XGBoost colapsou em teste apesar de AUC 0.74 — threshold calibrado em val foi otimista.

## Arquivos

| Arquivo                                | Descricao                                    |
|----------------------------------------|----------------------------------------------|
| `metricas.json`                        | Metricas no schema padrao do projeto         |
| `results_v4.json`                      | Relatorio completo original                  |
| `thresholds.json`                      | Thresholds P95 / P99 / Best-F1               |
| `xgb_threshold.json`                   | Threshold do classificador XGBoost           |
| `thresholds_per_feature_p95.npy`       | Thresholds P95 por feature                   |
| `thresholds_per_feature_p99.npy`       | Thresholds P99 por feature                   |
| `feature_masks.npz`                    | `info_mask`, `discriminative_mask`, `feature_auroc` |
| `feature_weights.npy`                  | Pesos por feature (derivados de AUROC)       |
| `feature_auroc.npy`                    | AUROC individual por feature                 |
| `features_selecionadas.json`           | Lista de features apos selecao               |
| `feature_analysis.png`                 | Figura: analise exploratoria de features     |
| `roc_pr_curves.png`                    | Figura: curvas ROC e PR                      |
| `xgb_feature_importance.png`           | Figura: importancia de features do XGBoost   |
