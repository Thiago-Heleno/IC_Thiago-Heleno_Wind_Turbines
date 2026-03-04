# Dataset usado

Gück, C., & Roelofs, C. (2024). Wind Turbine SCADA Data For Early Fault Detection (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10958775

https://zenodo.org/records/10958775

# Melhorias a fazer

- Remocao de outliers (mas tomar cuidado para nao perder informacao para deteccao de anomalias)
- Rodar em mais epocas, meu notebook nao aguenta
- Corrigir data leakage no notebook cnn_lstm_paper (mover split antes da normalizacao e feature selection)

# Metricas de teste nos notebooks

## wind_turbine_anomaly_detection.ipynb (CNN-LSTM Autoencoder v1)

| Threshold | Acuracia | Precisao | Recall | F1-Score |
|-----------|----------|----------|--------|----------|
| P95       | 0.8379   | 0.2572   | 0.4293 | 0.3217   |
| P99       | 0.8476   | 0.2148   | 0.2645 | 0.2371   |

## wind_turbine_anomaly_detection_v2.ipynb (CNN-LSTM Autoencoder v2)

AUC-ROC: 0.6591 | AUC-PR: 0.1833

| Threshold | Acuracia | Precisao | Recall | F1-Score |
|-----------|----------|----------|--------|----------|
| P95       | 0.7785   | 0.1598   | 0.3462 | 0.2186   |
| P99       | 0.8318   | 0.2084   | 0.3140 | 0.2505   |
| Best-F1   | 0.9074   | 0.0719   | 0.0029 | 0.0056   |

## wind_turbine_anomaly_detection_v3.ipynb (CNN-BiLSTM-Attention Autoencoder v3)

AUC-ROC: 0.7564 | AUC-PR: 0.2450

| Threshold         | Acuracia | Precisao | Recall | F1-Score |
|-------------------|----------|----------|--------|----------|
| Per-feature P95   | 0.1265   | 0.1090   | 1.0000 | 0.1966   |
| Per-feature P99   | 0.2208   | 0.1199   | 0.9922 | 0.2139   |
| Best-F1 (score)   | 0.5941   | 0.1797   | 0.7850 | 0.2925   |

## wind_turbine_cnn_lstm_paper.ipynb (CNN-LSTM Paper)

| Modelo   | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC-ROC |
|----------|--------------|---------------|------------|--------------|---------|
| CNN      | 73.37        | 79.09         | 84.60      | 81.75        | 0.69    |
| LSTM     | 37.40        | 66.67         | 22.38      | 33.51        | 0.55    |
| CNN-LSTM | 65.76        | 70.27         | 89.16      | 78.59        | 0.68    |

# Possiveis problemas

- **Data leakage no notebook cnn_lstm_paper**: MinMaxScaler, XGBoost feature selection e calculo de undersampling sao feitos sobre todos os dados antes do split train/val/test. A normalizacao e selecao de features deveriam ser feitas apenas no conjunto de treino.