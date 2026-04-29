# 05 — Autoencoder Induced Failure

> **Notebook:** [`notebooks/wind_turbine_autoencoder_induced_failure.ipynb`](../../notebooks/wind_turbine_autoencoder_induced_failure.ipynb)
> **Analise tecnica:** [`anotacoes/analise_autoencoder_induced_failure.md`](../../anotacoes/analise_autoencoder_induced_failure.md)
> **Metricas padronizadas:** [`metricas.json`](metricas.json)
> **Data de execucao:** 2026-04-29

## Objetivo

Autoencoder MLP semi-supervisionado calibrado com **falhas sinteticas** para derivar tres thresholds alternativos (Standard P95, Induced P95, Adaptive P50+gamma) e comparar trade-offs precision/recall/CARE.

## Pipeline

```
Wind Farm C (CARE)
    -> Split temporal CARE (train_test) + 80/20 calibracao
    -> Pre-processamento: Clipper + NanImputer + StandardScaler (so em normais)
    -> Selecao de features nao-supervisionada
    -> Optuna 30 trials AE (n_layers, code_size, lr, decay)
    -> Treino do AE so com normais
    -> Injecao 7% de falhas sinteticas em calibracao
    -> Optuna 30 trials gamma (FPR < 5%)
    -> 3 thresholds: Standard P95 | Induced P95 | Adaptive
    -> Avaliacao teste + CARE Score por evento
```

## Resultados principais

### Metricas por amostra (teste)

| Threshold | Valor | Precision | Recall | F1 | Accuracy |
|-----------|-------|-----------|--------|-----|----------|
| Standard (P95) | 0.6996 | 0.7136 | **0.1724** | **0.2777** | 0.5594 |
| **Induced (P95 induzido)** | 0.8436 | 0.7893 | 0.0704 | 0.1292 | 0.5340 |
| Adaptive (P50_ind+gamma) | 1.2539 | **0.8326** | 0.0205 | 0.0400 | 0.5167 |

### CARE Score (58 datasets, 27 anomalos / 31 normais)

| Threshold | Acc | EF1_2 | WS | **CARE** |
|-----------|-----|-------|-----|----------|
| Standard (P95) | 0.927 | 0.670 | 0.188 | 0.5424 |
| **Induced (P95 induzido)** ✓ | **0.980** | **0.680** | 0.086 | **0.5453** |
| Adaptive | 0.996 | 0.632 | 0.034 | 0.5315 |

**Melhor threshold por CARE: Induced (P95 induzido)** — combina alta especificidade em normais (Acc=0.98) com EF1_2 razoavel (0.68).

### Hiperparametros otimos (Optuna AE)

| HP | Valor |
|----|-------|
| n_layers | 1 |
| code_size | 57 |
| learning_rate | 2.28e-3 |
| decay_rate | 0.9704 |
| gamma (segunda otimizacao) | 0.7491 |

## Artefatos

| Arquivo | Conteudo |
|---------|----------|
| `autoencoder_induced.keras` | Modelo final treinado |
| `autoencoder_best.keras` | Checkpoint do melhor epoch |
| `best_params.json` | Hiperparametros Optuna |
| `threshold_params.json` | Tres thresholds + gamma |
| `metricas.json` | Metricas completas |
| `comparison.csv` | Comparacao dos tres thresholds |
| `care_results.csv` / `care_summary.csv` | CARE por dataset / agregado |
| `comparison_plots.png` | Loss, RMSE, serie temporal, CM, CARE |

## Como reproduzir

```bash
nohup conda run --no-capture-output -n theleno env \
  CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  MPLBACKEND=Agg \
  taskset -c 0-3 \
  python notebooks/wind_turbine_autoencoder_induced_failure.py \
  > notebooks/wind_turbine_autoencoder_induced_failure.txt 2>&1 &
```
