# Análise dos Resultados — Keras Autoencoder Pipeline

**Data:** 2026-04-01
**Branch:** updated
**Dataset:** 58 turbinas eólicas (51 anômalas, 7 normais)

---

## 1. Melhores Hiperparâmetros (Optuna)

```json
{
  "n_layers": 2,
  "code_size": 35,
  "learning_rate": 9.795e-4,
  "decay_rate": 0.9941,
  "gamma": 0.0761
}
```

---

## 2. Métricas Globais

| Métrica      | Valor     | Descrição                                      |
|--------------|-----------|------------------------------------------------|
| **CARE**     | **0.6993** | Score composto do framework de avaliação       |
| **EF1_β=2**  | **0.8989** | F1 a nível de evento (detecção de falhas)      |
| **WS**       | **0.6571** | Warning System score (detecção precoce)        |
| **F1_β=2**   | **0.3928** | F1 a nível de timestep                         |
| **Acc**      | **0.7738** | Acurácia em períodos normais                   |

---

## 3. Comparação com Versão Anterior (backup_v1)

| Métrica   | v1 (backup) | Atual      | Melhora    |
|-----------|-------------|------------|------------|
| CARE      | 0.0000      | **0.6993** | +0.6993    |
| EF1_β=2   | 0.0000      | **0.8989** | +0.8989    |
| WS        | 0.0040      | **0.6571** | +0.6531    |
| F1_β=2    | 0.0152      | **0.3928** | +0.3776    |
| Acc       | 1.0000      | 0.7738     | -0.2262    |

> **Nota:** O v1 era um modelo com viés "nunca alarma" — Acc perfeita porque
> nenhum alarme era disparado. O pipeline atual reverteu completamente esse
> comportamento, com ganho expressivo em todas as métricas de detecção.

---

## 4. Matriz de Confusão (nível de dataset)

|                   | Alarm = 1 | Alarm = 0 |
|-------------------|-----------|-----------|
| **has_anomaly = 1** | 48 (TP)  | 3 (FN)   |
| **has_anomaly = 0** | 6 (FP)   | 1 (TN)   |

| Medida         | Valor  |
|----------------|--------|
| Sensibilidade  | 94.1%  |
| Precisão       | 88.9%  |
| Especificidade | 14.3%  |

---

## 5. Falsos Negativos — Anomalias Não Detectadas (3 datasets)

| Dataset | WS    | F1_β=2 | criticality_max | Motivo provável                      |
|---------|-------|--------|-----------------|--------------------------------------|
| 80.csv  | 0.429 | 0.303  | 2               | Anomalia sutil, criticality baixo    |
| 81.csv  | 0.067 | 0.266  | 2               | Anomalia sutil, WS muito baixo       |
| 46.csv  | 0.625 | 0.370  | 1               | Criticality nunca ultrapassou limiar |

Os três casos têm `criticality_max` de 1-2 (limiar atual = 6). As anomalias são
reais mas sutis — o threshold adaptativo não dispara alarme.

---

## 6. Falsos Positivos — Alarmes em Dados Normais (6 datasets)

| Dataset | Acc   | criticality_max | Observação                                  |
|---------|-------|-----------------|---------------------------------------------|
| 85.csv  | 0.090 | 1302            | **Outlier severo** — possível data drift    |
| 32.csv  | 0.869 | 105             | Criticality alta, dados com ruído           |
| 37.csv  | 0.824 | 134             | Idem                                        |
| 29.csv  | 0.717 | 50              | Acc moderada                                |
| 8.csv   | 0.934 | 18              | Acc alta, criticality baixa                 |
| 62.csv  | 0.986 | 10              | Acc muito alta, criticality mínima          |

**85.csv** é um caso patológico — Acc=9% e criticality=1302 indicam que o
modelo está em distribuição completamente diferente do treinamento (data drift).
Os demais têm criticality moderada e podem ser mitigados ajustando o threshold.

---

## 7. Parâmetros de Threshold

```json
{
  "fixed_threshold_p95": 2.954,
  "fixed_threshold_fbeta": -0.613,
  "gamma": 0.4997,
  "adaptive_formula": "rmse_pred + gamma",
  "smoothing_window": 5,
  "criticality_threshold": 6
}
```

Score standardization: μ = 0.279, σ = 0.102

---

## 8. Principais Features (ARCANA)

As features mais recorrentes no top-5 de importância entre os datasets:
- `sensor_X_std` — desvio padrão de sensores
- `sensor_42/88/89/191/195` — sensores específicos com alta discriminabilidade
- `reactive_power_X_*_roll6_std` — potência reativa com janela rolling

---

## 9. Próximos Passos Sugeridos

1. **Investigar 85.csv** — verificar qualidade e distribuição dos dados
2. **Ajustar `criticality_threshold`** (6 → 8-10) para reduzir falsos positivos
3. **Explorar datasets 80, 81, 46** — verificar se mudança no gamma melhora recall
4. **Visualizações** — distribuição F1/WS por dataset, curvas ROC, análise ARCANA
