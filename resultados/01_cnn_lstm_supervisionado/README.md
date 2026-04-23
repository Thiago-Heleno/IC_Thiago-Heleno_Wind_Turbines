# 01 — CNN-LSTM Supervisionado

Pipeline **supervisionado** com tres arquiteturas treinadas em paralelo: CNN pura, LSTM pura, CNN-LSTM combinado.

- **Notebook:** [notebooks/wind_turbine_cnn_lstm_paper.ipynb](../../notebooks/wind_turbine_cnn_lstm_paper.ipynb)
- **Script:** [notebooks/wind_turbine_cnn_lstm_paper.py](../../notebooks/wind_turbine_cnn_lstm_paper.py)
- **Status:** pendente (artefatos gerados na primeira execucao)

## Ideia central

Aprender a distinguir janelas rotuladas como *normal* vs *anomalia*, com **split por evento** (sem vazamento temporal entre treino/val/teste) e undersampling aplicado **apenas no treino** para preservar a distribuicao natural em val/teste.

## Como rodar

```bash
jupyter nbconvert --to notebook --execute notebooks/wind_turbine_cnn_lstm_paper.ipynb
```

Saida cai nesta pasta. Preencher `metricas.json` com os numeros finais apos execucao.

## Arquivos esperados

| Arquivo                 | Descricao                                    |
|-------------------------|----------------------------------------------|
| `cnn_lstm_model.pth`    | Pesos do modelo hibrido                      |
| `cnn_model.pth`         | Pesos do modelo CNN puro                     |
| `lstm_model.pth`        | Pesos do modelo LSTM puro                    |
| `scaler.npz`            | Parametros do scaler MinMax para reproducao  |
| `results.json`          | Relatorio completo (splits, metricas, HPs)   |
| `metricas.json`         | Metricas no schema padrao do projeto         |
