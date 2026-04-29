# 04 — Classifier Induced Failure

> **Notebook:** [`notebooks/wind_turbine_classifier_induced_failure.ipynb`](../../notebooks/wind_turbine_classifier_induced_failure.ipynb)
> **Analise tecnica:** [`anotacoes/analise_classifier_induced_failure.md`](../../anotacoes/analise_classifier_induced_failure.md)
> **Metricas padronizadas:** [`metricas.json`](metricas.json)
> **Data de execucao:** 2026-04-29

## Objetivo

Classificador supervisionado MLP de 3 classes (Normal / Anomalia Real / Anomalia Induzida) para testar se **data augmentation com falhas sinteticas** melhora a deteccao de anomalias reais em SCADA.

## Pipeline

```
Wind Farm C (CARE)
    -> Split por evento 70/15/15 (sem leakage temporal)
    -> Pre-processamento: Clipper + NanImputer + StandardScaler
    -> Selecao de features nao-supervisionada
    -> Injecao 7% de falhas sinteticas (gaussian/drift/spike) no treino
    -> Optuna 30 trials (n_layers, hidden_units, dropout, lr)
    -> Treino MLP 3 classes + Treino baseline 2 classes
    -> Threshold via curva PR (max F1)
    -> Avaliacao teste + CARE Score por evento
```

## Resultados principais

### Metricas por amostra (teste)

| Modelo | Precision | Recall | F1 | Accuracy |
|--------|-----------|--------|-----|----------|
| **Modelo Induzido (3 classes)** | 0.5079 | **1.0000** | **0.6736** | 0.5079 |
| Baseline (2 classes) | 0.4073 | 0.6657 | 0.5054 | 0.3381 |

**Ganho da injecao sintetica:** F1 +16.8 pp (0.674 vs 0.505).

### CARE Score (10 datasets de teste)

| Modelo | EF1_2 | Acc | WS | **CARE** |
|--------|-------|-----|-----|----------|
| Induzido | 0.5556 | 3.6e-6 | 1.0000 | 3.6e-6 |
| Baseline | 0.5556 | 2.2e-5 | 0.6486 | 2.2e-5 |

CARE colapsa pela Acc=0 em eventos normais (ambos modelos alarmam em todos).

### Hiperparametros otimos

| HP | Valor |
|----|-------|
| n_layers | 2 |
| hidden_units | 192 |
| dropout_rate | 0.2945 |
| learning_rate | 5.47e-3 |

## Artefatos

| Arquivo | Conteudo |
|---------|----------|
| `classifier_induced.keras` | Modelo final treinado com falhas sinteticas |
| `classifier_best.keras` | Checkpoint do melhor epoch por val_loss |
| `best_params.json` | Hiperparametros Optuna |
| `metricas.json` | Metricas completas (teste + CARE) |
| `comparison.csv` | Tabela Induzido vs Baseline |
| `care_results.csv` | CARE por dataset |
| `comparison_plots.png` | Loss, matrizes confusao, ROC, PR, CARE |

## Como reproduzir

```bash
nohup conda run --no-capture-output -n theleno env \
  CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  MPLBACKEND=Agg \
  taskset -c 0-3 \
  python notebooks/wind_turbine_classifier_induced_failure.py \
  > notebooks/wind_turbine_classifier_induced_failure.txt 2>&1 &
```
