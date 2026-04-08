#!/usr/bin/env python
# coding: utf-8

# # Wind Turbine Anomaly Detection — CNN-BiLSTM-Attention Autoencoder + Melhorias v4
#
# ## Wind Farm C — SCADA Data
#
# Pipeline completo: treino do autoencoder (v3) + melhorias de deteccao sem retreino (v4).
#
# ### v4 Improvements (secoes 10-16)
# | Secao | Melhoria | Tipo |
# |-------|----------|------|
# | 10 | Analise de features informativas (AUROC por feature) | Diagnostico |
# | 11 | Score ponderado por AUROC + threshold Beta-F1 (beta=0.5) | Sem retreino |
# | 12 | Suavizacao temporal por majority vote (K=3/5/9/15) | Sem retreino |
# | 13 | Regra de contagem com mascara de features informativas | Sem retreino |
# | 14 | Classificador XGBoost pos-hoc nos erros de reconstrucao | Semi-supervisionado |
# | 15 | Avaliacao por evento (todas as estrategias) | Resultados |
# | 16 | Tabela comparativa v3 vs v4 | Resultados |
#
# Semi-supervised anomaly detection pipeline for offshore wind turbines using 10-min SCADA data.
# 
# - **Approach**: Autoencoder trained on healthy data only; anomalies detected via reconstruction error
# - **Architecture**: CNN -> BiLSTM -> Multi-Head Self-Attention -> LSTM Decoder
# - **Loss**: MAE (L1Loss)
# - **Feature selection**: Unsupervised (variance + operational correlation + redundancy removal)
# - **Detection**: Per-feature thresholds (P95/P99) + Best-F1 via PR curve
# 
# ### Pipeline
# 
# | Step | Description |
# |------|-------------|
# | 1 | Data loading & temporal split (70/15/15) |
# | 2 | Labeling (normal=0, anomaly=1) + healthy status filtering |
# | 3 | Min-Max normalization (healthy train data only) |
# | 4 | Unsupervised feature selection |
# | 5 | Sliding windows (36 steps = 6h) |
# | 6 | CNN-BiLSTM-Attention Autoencoder training (MAE) |
# | 7 | Per-feature threshold detection & evaluation |
# 
# ### Architecture Details (actual implementation)
# 
# | Component | Layers |
# |-----------|--------|
# | **Encoder CNN** | Conv1d(n_features->64, k=3) -> Conv1d(64->128, k=3) — preserves sequence length |
# | **Encoder BiLSTM** | BiLSTM(128->256 output) -> BiLSTM(256->128 output) |
# | **Attention** | MultiHeadAttention(embed=128, heads=4) + LayerNorm residual |
# | **Decoder LSTM** | LSTM(128->128) -> LSTM(128->128) — full sequence preserved (no mean pooling) |
# | **Decoder FC** | Linear(128->256) -> Linear(256->n_features) |
# | **Activation** | SELU throughout |
# 
# ### Laboratorio (maquina compartilhada)
# 
# - Antes de treinar: `nvidia-smi` — use a GPU indicada pelo professor.
# - Padroes neste notebook: GPU visivel `LAB_CUDA_DEVICE` ou `CUDA_VISIBLE_DEVICES` (padrao **1**), e **6** threads CPU via `LAB_CPU_THREADS`, se nao estiverem definidas no shell.
# - Dataset: pasta `CARE_To_Compare/Wind Farm C` na raiz do repositorio, ou variavel de ambiente `CARE_WIND_FARM_C` com o caminho completo dessa pasta.
# 

# In[ ]:


import os
from pathlib import Path

# --- Laboratorio (maquina compartilhada): definir ANTES de importar torch ---
# Padroes: segunda GPU e ate 6 threads. Sobrescreva no shell se o professor orientar.
_lab_threads = os.environ.get("LAB_CPU_THREADS", "6").strip()
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, _lab_threads)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("LAB_CUDA_DEVICE", "1")

import gc
import json
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, auc
)

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
try:
    torch.set_num_threads(max(1, int(_lab_threads)))
    torch.set_num_interop_threads(1)
except (ValueError, TypeError):
    pass

# Mantemos execucao em GPU quando disponivel (sem fallback automatico para CPU).
USE_CPU_FALLBACK = False
DEVICE = torch.device('cpu' if USE_CPU_FALLBACK else ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"Dispositivo: {DEVICE}")

if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
print(f"Recursos (lab): CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}  LAB_CPU_THREADS={_lab_threads}")

# --- Versoes dos pacotes ---
print(f"\nVersoes: torch={torch.__version__}, numpy={np.__version__}, "
      f"pandas={pd.__version__}, sklearn={__import__('sklearn').__version__}")

def _resolve_project_root() -> str:
    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        if (base / "README.md").is_file() and (base / "notebooks").is_dir():
            return str(base)
        if (base / "CARE_To_Compare").is_dir():
            return str(base)
    return str(cwd)

PROJECT_ROOT = _resolve_project_root()
_env_care = os.environ.get("CARE_WIND_FARM_C", "").strip()
if _env_care and os.path.isdir(os.path.join(_env_care, "datasets")):
    BASE_DIR = os.path.normpath(_env_care)
else:
    BASE_DIR = os.path.join(PROJECT_ROOT, "CARE_To_Compare", "Wind Farm C")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
EVENT_INFO_PATH = os.path.join(BASE_DIR, "event_info.csv")
FEATURE_DESC_PATH = os.path.join(BASE_DIR, "feature_description.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "resultados", "results_cnn_lstm_anomaly")
OUTPUT_DIR_V4 = OUTPUT_DIR  # script unificado: artefatos v3 e v4 no mesmo diretorio
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nCaminhos: PROJECT_ROOT={PROJECT_ROOT}\n  BASE_DIR={BASE_DIR}")

WINDOW_SIZE = 36
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.15
NUM_WORKERS = 4
CHECKPOINT_EVERY = 5  # salvar checkpoint a cada N epocas
# --- MIN_FEATURES_EXCEED: numero minimo de features que devem exceder o threshold
#     para classificar uma janela como anomalia.
#
#     RACIONAL: Com N features e threshold P95, o numero esperado de features
#     que excedem P95 por acaso em uma janela NORMAL e aprox. N * 0.05.
#     Ex: N=329 features -> ~16 features excedem P95 por acaso.
#     Usar MIN_FEATURES_EXCEED=1 classificaria QUASE TODA janela como anomalia,
#     gerando falsos positivos excessivos.
#
#     Calibre observando a distribuicao de n_exceeded nas janelas normais de validacao.
MIN_FEATURES_EXCEED_COEF = 0.05  # fracao de features esperadas acima do P95
MIN_FEATURES_EXCEED = 10  # anterior: 1 (causava FP excessivos com P95)
# Caminho para checkpoint a carregar (None = treinar do zero)
RESUME_FROM_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "resultados", "results_cnn_lstm_anomaly",
    "checkpoints", "checkpoint_epoch_100.pth"
)

# --- Feature selection parameters ---
VARIANCE_THRESHOLD = 1e-4        # Remove features com variancia < threshold
CORR_WITH_OPERATIONAL_MIN = 0.1  # Correlacao minima com variaveis operacionais
CORR_INTER_FEATURE_MAX = 0.95   # Remove features redundantes (corr > threshold)

FAST_MODE = False
FAST_MAX_EVENTS = 15
FAST_MAX_TRAIN_WINDOWS = 100_000

META_COLS = ['time_stamp', 'asset_id', 'id', 'train_test', 'status_type_id']
HEALTHY_STATUS = [0, 2]

# --- Mixed Precision (AMP) ---
USE_AMP = DEVICE.type == 'cuda'
if USE_AMP:
    print(f"\nMixed Precision (AMP): ativado — treinamento mais rapido e menor uso de VRAM")
else:
    print(f"\nMixed Precision (AMP): desativado (requer CUDA)")

print(f"\nJanela temporal: {WINDOW_SIZE} steps ({WINDOW_SIZE * 10 / 60:.0f} horas)")
print(f"Split: {TRAIN_RATIO*100:.0f}% treino / {VAL_RATIO*100:.0f}% validacao / {TEST_RATIO*100:.0f}% teste")
print(f"Selecao de features: nao-supervisionada (variancia + correlacao)")
print(f"  Variance threshold: {VARIANCE_THRESHOLD}")
print(f"  Correlacao minima c/ operacionais: {CORR_WITH_OPERATIONAL_MIN}")
print(f"  Correlacao max entre features: {CORR_INTER_FEATURE_MAX}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS} (patience={PATIENCE})")
print(f"Dropout: {DROPOUT_RATE}")
print(f"Loss: MAE (L1)")
print(f"MIN_FEATURES_EXCEED: {MIN_FEATURES_EXCEED}  # ~N*5%% para P95; calibre via validacao")
print(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")
print(f"Checkpoint: a cada {CHECKPOINT_EVERY} epocas")
if FAST_MODE:
    print(f"\n⚠️  FAST_MODE: ativado (max {FAST_MAX_EVENTS} CSVs, max {FAST_MAX_TRAIN_WINDOWS:,} janelas treino)")
    print(f"   Resultados NAO sao representativos do dataset completo.")
    print(f"   Para resultados finais, defina FAST_MODE = False.")
else:
    print(f"FAST_MODE: desativado (dados completos)")


# ## 1. Data Loading
# 
# Load 58 event CSVs from Wind Farm C. Data split is **temporal** (by timestamp) to prevent information leakage. Forward fill applied for missing values.

# In[ ]:


event_info = pd.read_csv(EVENT_INFO_PATH, sep=';')
event_info['event_start'] = pd.to_datetime(event_info['event_start'])
event_info['event_end'] = pd.to_datetime(event_info['event_end'])

print(f"Total de eventos: {len(event_info)}")
print(f"  Anomalias: {(event_info['event_label'] == 'anomaly').sum()}")
print(f"  Normais:   {(event_info['event_label'] == 'normal').sum()}")
event_info.head(10)


# In[ ]:


csv_files = sorted([f for f in os.listdir(DATASETS_DIR) if f.endswith('.csv')])
print(f"Total de arquivos CSV encontrados: {len(csv_files)}")

if FAST_MODE and FAST_MAX_EVENTS and len(csv_files) > FAST_MAX_EVENTS:
    csv_files = csv_files[:FAST_MAX_EVENTS]
    print(f"  FAST_MODE: limitado a {FAST_MAX_EVENTS} arquivos CSV")

print("Passo 1: coletando timestamps para definir cortes do split temporal...")
all_ts = []
for csv_file in csv_files:
    ts = pd.read_csv(os.path.join(DATASETS_DIR, csv_file),
                     sep=';', usecols=['time_stamp'])['time_stamp']
    all_ts.append(pd.to_datetime(ts))

all_ts = pd.concat(all_ts).sort_values().reset_index(drop=True)
n_total = len(all_ts)
ts_train_cutoff = all_ts.iloc[int(n_total * TRAIN_RATIO)]
ts_val_cutoff   = all_ts.iloc[int(n_total * (TRAIN_RATIO + VAL_RATIO))]
del all_ts; gc.collect()

print(f"  Amostras totais: {n_total:,}")
print(f"  Corte treino/val: {ts_train_cutoff}")
print(f"  Corte val/teste:  {ts_val_cutoff}")

sample_header = pd.read_csv(os.path.join(DATASETS_DIR, csv_files[0]), sep=';', nrows=0)
scols_all = [c for c in sample_header.columns if c not in META_COLS]
dtype_map = {c: np.float32 for c in scols_all}

print("\nPasso 2: processamento por arquivo (ffill + labeling + split)...")
anomaly_dict = {}
for _, row in event_info[event_info['event_label'] == 'anomaly'].iterrows():
    anomaly_dict[row['event_id']] = (row['event_start'], row['event_end'])

train_parts, val_parts, test_parts = [], [], []

for i, csv_file in enumerate(csv_files):
    event_id = int(csv_file.replace('.csv', ''))
    filepath = os.path.join(DATASETS_DIR, csv_file)

    df = pd.read_csv(filepath, sep=';', dtype=dtype_map)
    df['event_id'] = event_id
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])

    scols = [c for c in df.columns if c not in META_COLS + ['event_id']]
    df[scols] = df[scols].ffill(limit=6)  # limit=6: max 1h (6x10min) de propagacao
    df[scols] = df[scols].bfill(limit=6)
    n_nan = df[scols].isna().sum().sum()
    if n_nan > 0:
        print(f"  AVISO: evento {event_id} tem {n_nan} NaN apos ffill/bfill — preenchendo com 0")
        df[scols] = df[scols].fillna(0)

    df['label'] = 0
    if event_id in anomaly_dict:
        es, ee = anomaly_dict[event_id]
        mask_anom = (df['time_stamp'] >= es) & (df['time_stamp'] <= ee)
        df.loc[mask_anom, 'label'] = 1

    df['is_healthy'] = (
        (df['label'] == 0) & df['status_type_id'].isin(HEALTHY_STATUS)
    ).astype(np.int8)

    m_tr = df['time_stamp'] < ts_train_cutoff
    m_va = (df['time_stamp'] >= ts_train_cutoff) & (df['time_stamp'] < ts_val_cutoff)
    m_te = df['time_stamp'] >= ts_val_cutoff

    if m_tr.any(): train_parts.append(df.loc[m_tr])
    if m_va.any(): val_parts.append(df.loc[m_va])
    if m_te.any(): test_parts.append(df.loc[m_te])

    del df
    if (i + 1) % 10 == 0:
        gc.collect()
    if (i + 1) % 20 == 0 or (i + 1) == len(csv_files):
        print(f"  Processados: {i + 1}/{len(csv_files)}")

print("\nConcatenando e ordenando splits...")

df_train = pd.concat(train_parts, ignore_index=True)
del train_parts; gc.collect()
df_train.sort_values('time_stamp', inplace=True)
df_train.reset_index(drop=True, inplace=True)

df_val = pd.concat(val_parts, ignore_index=True)
del val_parts; gc.collect()
df_val.sort_values('time_stamp', inplace=True)
df_val.reset_index(drop=True, inplace=True)

df_test = pd.concat(test_parts, ignore_index=True)
del test_parts; gc.collect()
df_test.sort_values('time_stamp', inplace=True)
df_test.reset_index(drop=True, inplace=True)

n = len(df_train) + len(df_val) + len(df_test)
for name, sp in [('Treino', df_train), ('Validacao', df_val), ('Teste', df_test)]:
    ns = len(sp)
    mem = sp.memory_usage(deep=True).sum() / 1e6
    print(f"  {name:10s}: {ns:>7,} amostras ({ns/n*100:5.1f}%) | "
          f"Anomalias: {(sp['label']==1).sum():>5,} | "
          f"Saudaveis: {(sp['is_healthy']==1).sum():>7,} | "
          f"Mem: {mem:,.0f} MB")

# --- Verificar eventos que cruzam os pontos de corte do split ---
# Um evento que cruza um corte tera timesteps em dois splits distintos.
# Isso nao e um bug, mas e importante documentar para transparencia academica.
print("\nVerificacao de eventos cruzando cortes do split temporal:")
events_crossing_train_val = 0
events_crossing_val_test  = 0
for csv_file in csv_files:
    event_id = int(csv_file.replace('.csv', ''))
    ts_series = pd.to_datetime(
        pd.read_csv(
            os.path.join(DATASETS_DIR, csv_file), sep=';', usecols=['time_stamp']
        )['time_stamp']
    )
    has_train = (ts_series < ts_train_cutoff).any()
    has_val   = ((ts_series >= ts_train_cutoff) & (ts_series < ts_val_cutoff)).any()
    has_test  = (ts_series >= ts_val_cutoff).any()
    if has_train and has_val:
        events_crossing_train_val += 1
    if has_val and has_test:
        events_crossing_val_test += 1
print(f"  Eventos cruzando corte treino/val: {events_crossing_train_val}/{len(csv_files)}")
print(f"  Eventos cruzando corte val/teste:  {events_crossing_val_test}/{len(csv_files)}")
if events_crossing_train_val > 0 or events_crossing_val_test > 0:
    print("  NOTA: eventos cruzando cortes terao seus timesteps divididos entre splits.")
    print("  Isso e esperado no split temporal global e NAO configura vazamento de dados.")
    print("  Para splits estritamente por evento, use event_id como chave de divisao.")
else:
    print("  Nenhum evento cruza os cortes do split (splits completamente disjuntos).")

all_labels = pd.concat([df_train['label'], df_val['label'], df_test['label']])
all_healthy = pd.concat([df_train['is_healthy'], df_val['is_healthy'], df_test['is_healthy']])
all_status = pd.concat([df_train['status_type_id'], df_val['status_type_id'], df_test['status_type_id']])

print("Distribuicao de labels (todos os splits combinados):")
print(f"  Saudavel (0):  {(all_labels == 0).sum():,} "
      f"({(all_labels == 0).mean()*100:.1f}%)")
print(f"  Anomalia (1):  {(all_labels == 1).sum():,} "
      f"({(all_labels == 1).mean()*100:.1f}%)")
print(f"\nAmostras estritamente saudaveis (label=0 e status normal/idling): "
      f"{(all_healthy == 1).sum():,} ({(all_healthy == 1).mean()*100:.1f}%)")
print(f"\nDistribuicao de status_type_id:")
print(all_status.value_counts().sort_index().to_string())
del all_labels, all_healthy, all_status


# ## 3. Temporal Split
# 
# Strictly temporal split (70/15/15) by global timestamp ordering. Scaler and feature selection fitted on training data only.
# 
# > **NOTA**: O threshold `Best-F1` e calibrado usando o conjunto de validacao. Por isso, metricas de *validacao* com esse threshold sao otimisticamente enviesadas (o mesmo split e usado para ajuste e avaliacao). **Use exclusivamente as metricas do conjunto de TESTE como resultado final**.

# In[ ]:


n = len(df_train) + len(df_val) + len(df_test)

print("Verificacao da divisao temporal (ja realizada na etapa de carregamento):\n")
for name, split in [('Treino', df_train), ('Validacao', df_val), ('Teste', df_test)]:
    n_total = len(split)
    n_anom = (split['label'] == 1).sum()
    n_healthy = (split['is_healthy'] == 1).sum()
    print(f"{name:10s}: {n_total:>7,} amostras ({n_total/n*100:5.1f}%) | "
          f"Anomalias: {n_anom:>5,} | Saudaveis: {n_healthy:>7,} | "
          f"Periodo: {split['time_stamp'].min()} -- {split['time_stamp'].max()}")

print(f"\nCorte treino/val: {ts_train_cutoff}")
print(f"Corte val/teste:  {ts_val_cutoff}")


# ## 4. Normalization
# 
# Min-Max scaling to [0, 1]. Scaler fitted **only on healthy training samples** (`is_healthy=1`) to avoid anomaly contamination. Values outside [0, 1] in val/test are preserved (no clipping).

# In[ ]:


sensor_cols = [c for c in df_train.columns
              if c not in META_COLS + ['event_id', 'label', 'is_healthy']]
print(f"Total de features de sensores: {len(sensor_cols)}")

# Preencher NaN introduzido pelo pd.concat quando eventos têm colunas diferentes
for _name, _df in [('Treino', df_train), ('Validacao', df_val), ('Teste', df_test)]:
    _n = _df[sensor_cols].isna().sum().sum()
    if _n > 0:
        print(f"  AVISO: {_name} tem {int(_n)} NaN em sensor_cols apos concat — preenchendo com 0")
        _df[sensor_cols] = _df[sensor_cols].fillna(0)

# Ajustar scaler SOMENTE nas amostras saudaveis de treino
# (consistente com o paradigma semi-supervisionado do autoencoder)
df_train_healthy = df_train[df_train['is_healthy'] == 1]
print(f"Amostras saudaveis de treino para ajuste do scaler: {len(df_train_healthy):,} "
      f"({len(df_train_healthy)/len(df_train)*100:.1f}% do treino)")

scaler = MinMaxScaler(feature_range=(0, 1))
CHUNK_FIT = 50_000
for start in range(0, len(df_train_healthy), CHUNK_FIT):
    end = min(start + CHUNK_FIT, len(df_train_healthy))
    scaler.partial_fit(df_train_healthy[sensor_cols].iloc[start:end])

del df_train_healthy; gc.collect()
print("Scaler ajustado via partial_fit (somente amostras saudaveis de treino)")

# --- Aplicar normalizacao com scaler.transform() em chunks de memoria ---
# scaler.transform(X) equivale a X * scaler.scale_ + scaler.min_,
# onde scaler.min_ = -X_min * scaler.scale_ (atributo interno sklearn).
# Usar transform() e mais legivel, auditavel e evita bug de indexacao manual.
# Valores fora de [0,1] em val/teste sao esperados (extrapolar range do treino)
# e NAO sao clipados (preserva a informacao de anomalia).
CHUNK_TRANSFORM = 100_000

for split_name, df_split in [('Treino', df_train), ('Validacao', df_val), ('Teste', df_test)]:
    n_total_vals = len(df_split) * len(sensor_cols)
    n_oor = 0
    col_idx = df_split.columns.get_indexer(sensor_cols)

    for start in range(0, len(df_split), CHUNK_TRANSFORM):
        end = min(start + CHUNK_TRANSFORM, len(df_split))
        chunk_vals = df_split[sensor_cols].iloc[start:end].values.astype(np.float32)
        transformed = scaler.transform(chunk_vals).astype(np.float32)
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=1.0, neginf=0.0)
        df_split.iloc[start:end, col_idx] = transformed
        n_oor += int(((transformed < 0) | (transformed > 1)).sum())

    gc.collect()
    print(f"  {split_name} normalizado | Valores fora de [0,1]: {n_oor:,} "
          f"({n_oor/n_total_vals*100:.4f}%)")

for name, split in [('Treino', df_train), ('Validacao', df_val), ('Teste', df_test)]:
    vals = split[sensor_cols]
    print(f"  {name:10s} - min: {vals.min().min():.4f}, "
          f"max: {vals.max().max():.4f}, media: {vals.mean().mean():.4f}")

print("\nNormalizacao concluida (scaler.transform() em chunks, sem clipping)")


# ## 5. Unsupervised Feature Selection
# 
# Three-stage selection using **only healthy training data** (no labels used):
# 
# 1. **Variance filter**: remove near-zero variance features (< 1e-4)
# 2. **Operational correlation**: keep features with |r| >= 0.1 with wind speed, power, or rotor speed
# 3. **Redundancy removal**: among features with |r| > 0.95, keep the one with higher operational correlation

# In[ ]:


# ---- Selecao nao-supervisionada: somente dados saudaveis de treino ----
df_healthy_train = df_train[df_train['is_healthy'] == 1]
print(f"Amostras saudaveis de treino para selecao de features: {len(df_healthy_train):,}")

# --- Passo 1: Filtro de variancia ---
variances = df_healthy_train[sensor_cols].var()
high_var_mask = variances >= VARIANCE_THRESHOLD
cols_after_var = variances[high_var_mask].index.tolist()
n_removed_var = len(sensor_cols) - len(cols_after_var)
print(f"\n[Passo 1] Filtro de variancia (threshold={VARIANCE_THRESHOLD}):")
print(f"  Removidas: {n_removed_var} features com variancia ~0")
print(f"  Restantes: {len(cols_after_var)}")

# --- Passo 2: Correlacao com variaveis operacionais ---
# Identificar colunas operacionais presentes nos dados
operational_patterns = ['wind_speed', 'power_2', 'power_5', 'power_6',
                        'sensor_144', 'sensor_145']  # wind speeds, active powers, rotor speeds
operational_cols = []
for pat in operational_patterns:
    for col in cols_after_var:
        if col.startswith(pat + '_') or col == pat:
            operational_cols.append(col)
operational_cols = list(set(operational_cols))

if not operational_cols:
    # Fallback: buscar por padrao mais amplo
    for col in cols_after_var:
        if any(col.startswith(p) for p in ['wind_speed_', 'power_', 'sensor_144_', 'sensor_145_']):
            operational_cols.append(col)
    operational_cols = list(set(operational_cols))

print(f"\n[Passo 2] Correlacao com variaveis operacionais:")
print(f"  Variaveis operacionais encontradas: {len(operational_cols)}")
for oc in sorted(operational_cols):
    print(f"    - {oc}")

# Calcular correlacao de cada feature com as operacionais (amostrar se necessario)
MAX_CORR_SAMPLES = 100_000
if len(df_healthy_train) > MAX_CORR_SAMPLES:
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(df_healthy_train), MAX_CORR_SAMPLES, replace=False)
    df_corr_sample = df_healthy_train.iloc[idx]
else:
    df_corr_sample = df_healthy_train

# Features candidatas = todas apos filtro de variancia, MENOS as operacionais
candidate_cols = [c for c in cols_after_var if c not in operational_cols]

# Para cada candidata, calcular max |correlacao| com qualquer operacional
corr_scores = {}
for col in candidate_cols:
    max_abs_corr = 0.0
    for op_col in operational_cols:
        r = df_corr_sample[col].corr(df_corr_sample[op_col])
        if not np.isnan(r):
            max_abs_corr = max(max_abs_corr, abs(r))
    corr_scores[col] = max_abs_corr

# Sempre incluir as operacionais + candidatas com correlacao suficiente
cols_after_corr = list(operational_cols)  # operacionais sempre entram
for col, score in corr_scores.items():
    if score >= CORR_WITH_OPERATIONAL_MIN:
        cols_after_corr.append(col)

n_removed_corr = len(cols_after_var) - len(cols_after_corr)
print(f"  Removidas: {n_removed_corr} features com baixa correlacao operacional")
print(f"  Restantes: {len(cols_after_corr)}")

# --- Passo 3: Remocao de redundancia (correlacao inter-feature) ---
# Computar matriz de correlacao entre features selecionadas
corr_matrix = df_corr_sample[cols_after_corr].corr().abs()

# Greedy: para cada par com |r| > threshold, remover a com menor corr operacional media
cols_to_drop = set()
corr_cols = list(cols_after_corr)
for i in range(len(corr_cols)):
    if corr_cols[i] in cols_to_drop:
        continue
    for j in range(i + 1, len(corr_cols)):
        if corr_cols[j] in cols_to_drop:
            continue
        if corr_matrix.loc[corr_cols[i], corr_cols[j]] > CORR_INTER_FEATURE_MAX:
            # Manter a feature com maior score operacional
            score_i = corr_scores.get(corr_cols[i], 1.0)  # operacionais recebem 1.0
            score_j = corr_scores.get(corr_cols[j], 1.0)
            if score_i >= score_j:
                cols_to_drop.add(corr_cols[j])
            else:
                cols_to_drop.add(corr_cols[i])

selected_features = [c for c in cols_after_corr if c not in cols_to_drop]
selected_features.sort()

print(f"\n[Passo 3] Remocao de redundancia (threshold={CORR_INTER_FEATURE_MAX}):")
print(f"  Removidas: {len(cols_to_drop)} features redundantes")
print(f"  Features finais selecionadas: {len(selected_features)}")

# --- Resumo ---
print(f"\n{'='*60}")
print(f"RESUMO DA SELECAO DE FEATURES")
print(f"{'='*60}")
print(f"  Features originais:         {len(sensor_cols)}")
print(f"  Apos filtro de variancia:   {len(cols_after_var)} (-{n_removed_var})")
print(f"  Apos filtro de correlacao:  {len(cols_after_corr)} (-{n_removed_corr})")
print(f"  Apos remocao redundancia:   {len(selected_features)} (-{len(cols_to_drop)})")
print(f"  Reducao total:              {len(sensor_cols) - len(selected_features)} features removidas "
      f"({(1 - len(selected_features)/len(sensor_cols))*100:.1f}%)")

# Salvar ranking para analise
feature_ranking = pd.DataFrame({
    'feature': list(corr_scores.keys()),
    'max_abs_corr_operational': list(corr_scores.values()),
    'variance': [variances[c] for c in corr_scores.keys()],
    'selected': [c in selected_features for c in corr_scores.keys()]
}).sort_values('max_abs_corr_operational', ascending=False).reset_index(drop=True)

print(f"\nTop 20 features por correlacao com operacionais:")
print(feature_ranking.head(20).to_string())

del df_healthy_train, df_corr_sample, corr_matrix; gc.collect()

# --- Recalcular MIN_FEATURES_EXCEED com base no numero real de features selecionadas ---
# P95 gera ~5% de falsos positivos por feature; logo usamos N*0.05 como limiar minimo.
# Este valor e calibravel via curva PR no conjunto de validacao (veja celula de avaliacao).
MIN_FEATURES_EXCEED = max(1, int(len(selected_features) * MIN_FEATURES_EXCEED_COEF))
print(f"\nMIN_FEATURES_EXCEED (dinamico): {MIN_FEATURES_EXCEED} "
      f"= max(1, int({len(selected_features)} * {MIN_FEATURES_EXCEED_COEF}))")
print(f"  Interpretacao: janela classificada como anomalia se >= {MIN_FEATURES_EXCEED} "
      f"features excederem o threshold P95/P99.")
print(f"  Isso corresponde a ~{MIN_FEATURES_EXCEED_COEF*100:.0f}% das {len(selected_features)} "
      f"features, controlando FP aleatorios do P95.")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Grafico 1: Top 20 features por variancia
top20_var = feature_ranking.sort_values('variance', ascending=False).head(20)
colors_var = ['#55A868' if s else '#C44E52' for s in top20_var['selected']]
axes[0].barh(range(len(top20_var)), top20_var['variance'].values,
             color=colors_var, edgecolor='black', linewidth=0.5)
axes[0].set_yticks(range(len(top20_var)))
axes[0].set_yticklabels(top20_var['feature'].values, fontsize=8)
axes[0].axvline(x=VARIANCE_THRESHOLD, color='red', linestyle='--', linewidth=1.5,
                label=f'Limiar = {VARIANCE_THRESHOLD}')
axes[0].set_xlabel('Variancia')
axes[0].set_title('Top 20 Features por Variancia (verde=selecionada, vermelho=removida)',
                  fontsize=11, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# Grafico 2: Top 20 features por correlacao operacional
top20 = feature_ranking.head(20)
colors_sel = ['#55A868' if s else '#C44E52' for s in top20['selected']]
axes[1].barh(range(len(top20)), top20['max_abs_corr_operational'].values,
             color=colors_sel, edgecolor='black', linewidth=0.5)
axes[1].set_yticks(range(len(top20)))
axes[1].set_yticklabels(top20['feature'].values, fontsize=8)
axes[1].axvline(x=CORR_WITH_OPERATIONAL_MIN, color='red', linestyle='--', linewidth=1.5,
                label=f'Limiar = {CORR_WITH_OPERATIONAL_MIN}')
axes[1].set_xlabel('Max |Correlacao| com Operacionais')
axes[1].set_title('Top 20 Features (verde=selecionada, vermelho=removida)', fontsize=11, fontweight='bold')
axes[1].invert_yaxis()
axes[1].legend(fontsize=9)
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()


# In[ ]:


keep_cols = ['time_stamp', 'event_id', 'label', 'is_healthy']
df_train_sel = df_train[keep_cols + selected_features].copy()
df_val_sel = df_val[keep_cols + selected_features].copy()
df_test_sel = df_test[keep_cols + selected_features].copy()

del df_train, df_val, df_test
gc.collect()

print("Dimensao apos selecao de features:")
print(f"  Treino:    {df_train_sel.shape}")
print(f"  Validacao: {df_val_sel.shape}")
print(f"  Teste:     {df_test_sel.shape}")


# ## 6. Sliding Windows
# 
# Windows of 36 timesteps (6h) built per event. A window is **healthy** if all 36 timesteps have `is_healthy=1`. A window is **anomalous** if any timestep has `label=1`.

# In[ ]:


def build_window_index(df, window_size, feature_cols):
    """Build per-group flat arrays and window metadata without materializing windows.

    Instead of storing an (N, W, F) tensor (~91 GB for training), we keep the
    flat feature arrays per event_id and a compact index of (eid, start, label,
    healthy) per window.  Windows are constructed on-the-fly in WindowDataset.
    """
    group_arrays = {}
    all_eid, all_start, all_label, all_healthy = [], [], [], []

    for eid, group in df.groupby('event_id'):
        group = group.sort_values('time_stamp').reset_index(drop=True)
        group_arrays[eid] = group[feature_cols].values.astype(np.float32)
        labels  = group['label'].values.astype(np.int8)
        healthy = group['is_healthy'].values.astype(np.int8)

        n = len(group)
        if n < window_size:
            continue

        nw = n - window_size + 1
        cum_l = np.concatenate([[0], np.cumsum(labels)])
        cum_h = np.concatenate([[0], np.cumsum(healthy)])
        w_labels  = (cum_l[window_size:] - cum_l[:nw] > 0).astype(np.int8)
        w_healthy = (cum_h[window_size:] - cum_h[:nw] == window_size).astype(np.int8)

        all_eid.append(np.full(nw, eid, dtype=np.int32))
        all_start.append(np.arange(nw, dtype=np.int32))
        all_label.append(w_labels)
        all_healthy.append(w_healthy)

    return (group_arrays,
            np.concatenate(all_eid),
            np.concatenate(all_start),
            np.concatenate(all_label),
            np.concatenate(all_healthy))


class WindowDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset: builds each window on-the-fly from flat arrays."""
    def __init__(self, group_arrays, eids, starts, window_size):
        self.ga     = group_arrays
        self.eids   = eids
        self.starts = starts
        self.ws     = window_size

    def __len__(self):
        return len(self.eids)

    def __getitem__(self, idx):
        e = int(self.eids[idx])
        s = int(self.starts[idx])
        return torch.from_numpy(self.ga[e][s:s + self.ws].copy()).to(dtype=torch.float32)


ga_train, eid_tr, start_tr, y_train, h_train = build_window_index(
    df_train_sel, WINDOW_SIZE, selected_features)
del df_train_sel; gc.collect()

ga_val, eid_va, start_va, y_val, h_val = build_window_index(
    df_val_sel, WINDOW_SIZE, selected_features)
del df_val_sel; gc.collect()

ga_test, eid_te, start_te, y_test, h_test = build_window_index(
    df_test_sel, WINDOW_SIZE, selected_features)
del df_test_sel; gc.collect()

n_features = len(selected_features)
print(f"Indices de janelas criados (window_size={WINDOW_SIZE}, features={n_features}):")
for name, eid_arr, y_arr, h_arr in [
    ('Treino', eid_tr, y_train, h_train),
    ('Validacao', eid_va, y_val, h_val),
    ('Teste', eid_te, y_test, h_test)]:
    print(f"  {name:10s}: {len(eid_arr):>7,} janelas | "
          f"Anomalias: {y_arr.sum():,} | Saudaveis: {h_arr.sum():,}")

ga_mem = sum(a.nbytes for a in ga_train.values()) / 1e9
print(f"\nMemoria: ~{ga_mem:.1f} GB (treino) vs ~{ga_mem * WINDOW_SIZE:.0f} GB "
      f"se janelas fossem materializadas")


# In[ ]:


healthy_mask_tr = (h_train == 1)
healthy_mask_va = (h_val == 1)

n_healthy_tr = int(healthy_mask_tr.sum())
n_healthy_va = int(healthy_mask_va.sum())

if FAST_MODE and FAST_MAX_TRAIN_WINDOWS and n_healthy_tr > FAST_MAX_TRAIN_WINDOWS:
    rng = np.random.RandomState(SEED)
    healthy_indices_all = np.where(healthy_mask_tr)[0]
    subsample_idx = rng.choice(len(healthy_indices_all), FAST_MAX_TRAIN_WINDOWS, replace=False)
    subsample_idx.sort()
    fast_mask = np.zeros(len(h_train), dtype=bool)
    fast_mask[healthy_indices_all[subsample_idx]] = True
    healthy_mask_tr = fast_mask
    n_healthy_tr_original = n_healthy_tr
    n_healthy_tr = FAST_MAX_TRAIN_WINDOWS
    print(f"FAST_MODE: sub-amostradas {n_healthy_tr:,} de {n_healthy_tr_original:,} janelas saudaveis de treino")

print("Janelas saudaveis para treinamento do autoencoder:")
print(f"  Treino:    ({n_healthy_tr}, {WINDOW_SIZE}, {n_features})")
print(f"  Validacao: ({n_healthy_va}, {WINDOW_SIZE}, {n_features})")

print(f"\nDimensoes de entrada do modelo:")
print(f"  Timesteps:  {WINDOW_SIZE}")
print(f"  Features:   {n_features}")


# ## 7. CNN-BiLSTM-Attention Autoencoder
# 
# | Component | Detail |
# |-----------|--------|
# | **Encoder CNN** | Conv1d(n_features->64, k=3, pad=1) -> Conv1d(64->128, k=3, pad=1) — preserves sequence length |
# | **Encoder BiLSTM** | BiLSTM(128->256 output) -> BiLSTM(256->128 output) |
# | **Attention** | MultiHeadAttention(embed_dim=128, heads=4) + LayerNorm residual connection |
# | **Decoder LSTM** | LSTM(128->128) -> LSTM(128->128) — full temporal sequence preserved (no mean pooling) |
# | **Decoder FC** | Linear(128->256) -> Linear(256->n_features) |
# | **Activation** | SELU (Kaiming-normal init approximating LeCun normal) |
# | **Loss** | MAE (L1Loss) — robust to outlier timesteps |
# | **Training** | Healthy windows only (`is_healthy == 1`) |
# 
# > **Note**: The full temporal sequence is preserved through encoder and decoder.
# > Anomaly discriminability depends on per-feature reconstruction MAE.
# > Features with near-zero variance in training may show flat reconstructions (predict mean),
# > which is expected behavior for low-information sensors.
# 

# In[ ]:


class CNNBiLSTMAttentionAutoencoder(nn.Module):
    def __init__(self, n_features, timesteps, dropout_rate=0.15, n_heads=4):
        super().__init__()
        self.n_features = n_features
        self.timesteps = timesteps

        # ---------- Encoder CNN ----------
        self.enc_conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)  # padding=1 preserva comprimento
        self.enc_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        # ---------- Encoder BiLSTM ----------
        self.enc_lstm1 = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.enc_lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        # saida: (batch, timesteps, 128)

        # ---------- Attention ----------
        self.attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=n_heads, batch_first=True, dropout=dropout_rate
        )
        self.attn_norm = nn.LayerNorm(128)

        # ---------- Decoder ----------
        # SEM mean pooling — a sequencia temporal completa passa ao decoder
        self.dec_lstm1 = nn.LSTM(128, 128, batch_first=True)
        self.dec_lstm2 = nn.LSTM(128, 128, batch_first=True)

        self.dropout = nn.Dropout(dropout_rate)
        # FC com capacidade suficiente para reconstruir n_features
        self.dec_fc1 = nn.Linear(128, 256)
        self.dec_fc2 = nn.Linear(256, n_features)

        self.selu = nn.SELU()

        # Lecun normal init
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, timesteps, n_features)

        # --- Encoder CNN (sem pooling agressivo para preservar timesteps) ---
        h = x.permute(0, 2, 1)          # (batch, n_features, timesteps)
        h = self.selu(self.enc_conv1(h)) # (batch, 64, timesteps)
        h = self.selu(self.enc_conv2(h)) # (batch, 128, timesteps)
        h = h.permute(0, 2, 1)          # (batch, timesteps, 128)

        # --- Encoder BiLSTM ---
        h, _ = self.enc_lstm1(h)         # (batch, timesteps, 256)
        h, _ = self.enc_lstm2(h)         # (batch, timesteps, 128)

        # --- Self-Attention ---
        attn_out, _ = self.attention(h, h, h)
        h = self.attn_norm(h + attn_out) # (batch, timesteps, 128)
        # NOTA: NAO fazemos mean pooling! A sequencia temporal eh preservada.

        # --- Decoder LSTM ---
        h, _ = self.dec_lstm1(h)         # (batch, timesteps, 128)
        h = self.selu(h)
        h, _ = self.dec_lstm2(h)         # (batch, timesteps, 128)
        h = self.selu(h)

        # --- Decoder FC (capacidade suficiente) ---
        h = self.dropout(h)
        h = self.selu(self.dec_fc1(h))   # (batch, timesteps, 256)
        h = self.dec_fc2(h)              # (batch, timesteps, n_features)

        return h


model = CNNBiLSTMAttentionAutoencoder(n_features, WINDOW_SIZE, dropout_rate=DROPOUT_RATE).to(DEVICE)
print(model)
print(f"\nParametros treinaveis: "
      f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Dropout rate: {DROPOUT_RATE}")


# ## 8. Training
# 
# - **Loss**: MAE (L1Loss) — robust to outliers
# - **Optimizer**: Adam (lr=1e-4)
# - **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
# - **Early stopping**: patience=5 on val_loss

# In[ ]:


train_dataset = WindowDataset(ga_train, eid_tr[healthy_mask_tr],
                              start_tr[healthy_mask_tr], WINDOW_SIZE)
val_dataset = WindowDataset(ga_val, eid_va[healthy_mask_va],
                            start_va[healthy_mask_va], WINDOW_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'),
                          persistent_workers=(NUM_WORKERS > 0))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'),
                        persistent_workers=(NUM_WORKERS > 0))

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# Mixed Precision
grad_scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

train_losses = []
val_losses = []
lr_history = []
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

# Diretorio para checkpoints periodicos
checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

_skip_training = False
if RESUME_FROM_CHECKPOINT and os.path.isfile(RESUME_FROM_CHECKPOINT):
    print(f"Carregando checkpoint: {RESUME_FROM_CHECKPOINT}")
    ckpt = torch.load(RESUME_FROM_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    train_losses = ckpt.get('train_losses', [])
    val_losses = ckpt.get('val_losses', [])
    lr_history = ckpt.get('lr_history', [])
    best_val_loss = min(val_losses) if val_losses else float('inf')
    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    print(f"  Epoca: {ckpt['epoch']} | Melhor val_loss registrado: {best_val_loss:.6f}")
    print("  Treinamento pulado — usando modelo do checkpoint.")
    _skip_training = True
else:
    print(f"Iniciando treinamento: {len(train_dataset):,} amostras treino, "
          f"{len(val_dataset):,} amostras validacao")
    print(f"Batch size: {BATCH_SIZE} | LR inicial: {LEARNING_RATE}")
    print(f"AMP: {'ativado' if USE_AMP else 'desativado'} | Workers: {NUM_WORKERS}")
    print(f"Checkpoint a cada {CHECKPOINT_EVERY} epocas em {checkpoint_dir}\n")

training_start_time = time.time()

for epoch in range(EPOCHS):
    if _skip_training:
        break
    epoch_start_time = time.time()
    model.train()
    epoch_train_loss = 0.0
    n_train_used = 0

    for batch_idx, batch_x in enumerate(train_loader):
        batch_x = batch_x.to(DEVICE, dtype=torch.float32, non_blocking=(DEVICE.type == 'cuda'))

        if not torch.isfinite(batch_x).all():
            print(f"[Treino] Batch {batch_idx}: NaN/Inf detectado no input, batch ignorado.")
            continue

        optimizer.zero_grad(set_to_none=True)

        try:
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                output = model(batch_x)
                loss = criterion(output, batch_x)

            if not torch.isfinite(loss):
                print(f"[Treino] Batch {batch_idx}: loss invalida ({loss.item()}), batch ignorado.")
                continue

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

        except RuntimeError as e:
            if DEVICE.type == 'cuda' and 'CUDA' in str(e).upper():
                raise RuntimeError(
                    "Erro CUDA durante treinamento. "
                    "Tente reduzir BATCH_SIZE (ex.: 32), reiniciar kernel e limpar cache com torch.cuda.empty_cache(). "
                    "Mensagem original: " + str(e)
                ) from e
            raise

        batch_size_eff = batch_x.size(0)
        epoch_train_loss += loss.item() * batch_size_eff
        n_train_used += batch_size_eff

    if n_train_used == 0:
        raise RuntimeError("Nenhum batch valido no treino (todos continham NaN/Inf ou falharam).")
    epoch_train_loss /= n_train_used

    model.eval()
    epoch_val_loss = 0.0
    n_val_used = 0
    with torch.no_grad():
        for batch_idx, batch_x in enumerate(val_loader):
            batch_x = batch_x.to(DEVICE, dtype=torch.float32, non_blocking=(DEVICE.type == 'cuda'))

            if not torch.isfinite(batch_x).all():
                print(f"[Validacao] Batch {batch_idx}: NaN/Inf detectado no input, batch ignorado.")
                continue

            with torch.amp.autocast('cuda', enabled=USE_AMP):
                output = model(batch_x)
                loss = criterion(output, batch_x)
            if not torch.isfinite(loss):
                print(f"[Validacao] Batch {batch_idx}: loss invalida, batch ignorado.")
                continue

            batch_size_eff = batch_x.size(0)
            epoch_val_loss += loss.item() * batch_size_eff
            n_val_used += batch_size_eff

    if n_val_used == 0:
        raise RuntimeError("Nenhum batch valido na validacao (todos continham NaN/Inf ou loss invalida).")
    epoch_val_loss /= n_val_used

    current_lr = optimizer.param_groups[0]['lr']
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    lr_history.append(current_lr)

    scheduler.step(epoch_val_loss)
    new_lr = optimizer.param_groups[0]['lr']

    marker = ''
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        marker = ' << melhor'
    else:
        patience_counter += 1

    # Tempo por epoca e estimativa de conclusao
    epoch_time = time.time() - epoch_start_time
    elapsed = time.time() - training_start_time
    epochs_done = epoch + 1
    remaining_epochs = EPOCHS - epochs_done
    eta_seconds = (elapsed / epochs_done) * remaining_epochs
    eta_min = eta_seconds / 60

    lr_msg = f' (LR reduzido: {current_lr:.2e} -> {new_lr:.2e})' if new_lr < current_lr else ''
    print(f"Epoca {epochs_done:02d}/{EPOCHS} | "
          f"Train: {epoch_train_loss:.6f} | "
          f"Val: {epoch_val_loss:.6f} | "
          f"LR: {current_lr:.2e} | "
          f"{epoch_time:.0f}s/ep | "
          f"ETA: {eta_min:.0f}min{marker}{lr_msg}")

    # Checkpoint periodico
    if epochs_done % CHECKPOINT_EVERY == 0:
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epochs_done:03d}.pth')
        torch.save({
            'epoch': epochs_done,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': grad_scaler.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'lr_history': lr_history,
        }, ckpt_path)
        print(f"  Checkpoint salvo: {ckpt_path}")

    if patience_counter >= PATIENCE:
        print(f"\nEarlyStopping na epoca {epochs_done} (paciencia={PATIENCE})")
        break

if best_model_state is None:
    raise RuntimeError("Treinamento finalizou sem salvar best_model_state (verifique estabilidade dos batches).")

model.load_state_dict(best_model_state)
if not _skip_training:
    total_time = time.time() - training_start_time
    print(f"\nMelhor val_loss: {best_val_loss:.6f}")
    print(f"Tempo total de treinamento: {total_time/60:.1f} min ({total_time/3600:.1f} h)")
else:
    print(f"\nMelhor val_loss (do checkpoint): {best_val_loss:.6f}")


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.plot(range(1, len(train_losses) + 1), train_losses, 'b-o',
        markersize=3, label='Treino')
ax1.plot(range(1, len(val_losses) + 1), val_losses, 'r-o',
        markersize=3, label='Validacao')
ax1.set_xlabel('Epoca')
ax1.set_ylabel('Loss (MAE)')
ax1.set_title('Curva de Treinamento -- CNN-BiLSTM-Attention Autoencoder')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, len(lr_history) + 1), lr_history, 'g-o', markersize=3)
ax2.set_xlabel('Epoca')
ax2.set_ylabel('Learning Rate')
ax2.set_title('Historico do Learning Rate (ReduceLROnPlateau)')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ## 9. Anomaly Detection
# 
# Per-feature reconstruction error (MAE) with individual thresholds:
# 
# - **P95/P99**: percentiles from healthy validation windows
# - **Classification**: anomaly if >= `MIN_FEATURES_EXCEED` features exceed their threshold
# - **Anomaly score**: `max(error_i / threshold_i)` — continuous score for AUC metrics

# In[ ]:


def compute_reconstruction_errors(model, dataset, per_feature_thresholds=None, batch_size=256):
    """Calcula MAE por feature e anomaly score por janela.

    Returns:
        pf_errors: (n_windows, n_features) -- MAE por feature (media sobre timesteps)
        anomaly_scores: (n_windows,) -- max(error_i / threshold_i) se thresholds fornecidos,
                        senao media global do MAE
    """
    model.eval()
    candidate_batches = [batch_size]
    for b in [128, 64, 32]:
        if b < batch_size and b not in candidate_batches:
            candidate_batches.append(b)

    last_error = None
    for bs in candidate_batches:
        try:
            pf_errors_list = []
            loader = DataLoader(dataset, batch_size=bs, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'),
                                persistent_workers=(NUM_WORKERS > 0))

            with torch.no_grad():
                for batch_x in loader:
                    batch_x = batch_x.to(DEVICE, dtype=torch.float32, non_blocking=(DEVICE.type == 'cuda'))
                    # Clampar para range seguro de float16 antes do autocast.
                    # float16 max ≈ 65504; valores maiores viram +Inf no autocast
                    # e propagam NaN pelo LSTM. Ocorre com outliers extremos no
                    # test set (ex: sensor com max normalizado de ~3.9M).
                    if USE_AMP:
                        batch_x = batch_x.clamp(min=-65504, max=65504)
                    with torch.amp.autocast('cuda', enabled=False):  # float32 na inferencia: evita overflow com outliers extremos no test set
                        output = model(batch_x)
                    pf_mae = (output - batch_x).abs().mean(dim=1)
                    if not torch.isfinite(pf_mae).all():
                        n_nan_in = (~torch.isfinite(batch_x)).sum().item()
                        n_nan_out = (~torch.isfinite(pf_mae)).sum().item()
                        raise RuntimeError(
                            f"MAE por feature retornou NaN/Inf durante inferencia. "
                            f"NaN/Inf na entrada: {n_nan_in}, na saida MAE: {n_nan_out}."
                        )
                    pf_errors_list.append(pf_mae.cpu().numpy())

            if len(pf_errors_list) == 0:
                raise RuntimeError("Dataset vazio durante inferencia.")

            pf_errors = np.vstack(pf_errors_list)
            if per_feature_thresholds is not None:
                anomaly_scores = (pf_errors / (per_feature_thresholds + 1e-10)).max(axis=1)
            else:
                anomaly_scores = pf_errors.mean(axis=1)
            return pf_errors, anomaly_scores

        except RuntimeError as e:
            last_error = e
            if DEVICE.type == 'cuda' and ('out of memory' in str(e).lower() or 'cuda' in str(e).lower()):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

    raise RuntimeError(
        f"Falha na inferencia para batches {candidate_batches}. Ultimo erro: {last_error}"
    )


ds_train_h  = WindowDataset(ga_train, eid_tr[healthy_mask_tr],
                            start_tr[healthy_mask_tr], WINDOW_SIZE)
ds_val_h    = WindowDataset(ga_val, eid_va[healthy_mask_va],
                            start_va[healthy_mask_va], WINDOW_SIZE)
ds_val_all  = WindowDataset(ga_val, eid_va, start_va, WINDOW_SIZE)
ds_test_all = WindowDataset(ga_test, eid_te, start_te, WINDOW_SIZE)

pf_errors_train_h, _ = compute_reconstruction_errors(model, ds_train_h)
pf_errors_val_h, _   = compute_reconstruction_errors(model, ds_val_h)

print("Erros de reconstrucao por feature (MAE) -- janelas saudaveis:")
print(f"  Treino    - media global: {pf_errors_train_h.mean():.6f}, "
      f"std global: {pf_errors_train_h.mean(axis=0).std():.6f}, "
      f"max feature media: {pf_errors_train_h.mean(axis=0).max():.6f}")
print(f"  Validacao - media global: {pf_errors_val_h.mean():.6f}, "
      f"std global: {pf_errors_val_h.mean(axis=0).std():.6f}, "
      f"max feature media: {pf_errors_val_h.mean(axis=0).max():.6f}")


# In[ ]:


thresholds_pf_p95 = np.percentile(pf_errors_val_h, 95, axis=0)
thresholds_pf_p99 = np.percentile(pf_errors_val_h, 99, axis=0)

print(f"Thresholds per-feature (baseados nas janelas saudaveis de validacao):")
print(f"  P95 -- min: {thresholds_pf_p95.min():.6f}, "
      f"media: {thresholds_pf_p95.mean():.6f}, "
      f"max: {thresholds_pf_p95.max():.6f}")
print(f"  P99 -- min: {thresholds_pf_p99.min():.6f}, "
      f"media: {thresholds_pf_p99.mean():.6f}, "
      f"max: {thresholds_pf_p99.max():.6f}")
print(f"  Shape: ({len(thresholds_pf_p95)},) features")

pf_errors_val_all, scores_val_all = compute_reconstruction_errors(
    model, ds_val_all, per_feature_thresholds=thresholds_pf_p95)
pf_errors_test_all, scores_test_all = compute_reconstruction_errors(
    model, ds_test_all, per_feature_thresholds=thresholds_pf_p95)

print(f"\nAnomaly scores (max-ratio com P95):")
print(f"  Validacao - media: {scores_val_all.mean():.4f}, max: {scores_val_all.max():.4f}")
print(f"  Teste     - media: {scores_test_all.mean():.4f}, max: {scores_test_all.max():.4f}")


# In[ ]:


precisions_pr, recalls_pr, thresholds_pr = precision_recall_curve(y_val, scores_val_all)
f1_scores_pr = 2 * (precisions_pr * recalls_pr) / (precisions_pr + recalls_pr + 1e-8)

if len(thresholds_pr) == 0:
    raise RuntimeError("precision_recall_curve retornou thresholds vazio. Verifique scores_val_all.")

# thresholds_pr tem tamanho N-1 em relacao a precisions/recalls.
best_f1_idx = int(np.argmax(f1_scores_pr[:-1]))
threshold_best_f1_score = float(thresholds_pr[best_f1_idx])

print("Threshold otimizado via curva Precision-Recall (no anomaly score de validacao):")
print(f"  Threshold Best-F1 (score): {threshold_best_f1_score:.4f}")
print(f"  F1 no ponto otimo: {f1_scores_pr[best_f1_idx]:.4f}")
print(f"  Precisao: {precisions_pr[best_f1_idx]:.4f}")
print(f"  Recall:   {recalls_pr[best_f1_idx]:.4f}")

print(f"\nResumo dos thresholds:")
print(f"  Per-feature P95: vetor ({len(thresholds_pf_p95)},) -- deteccao via MIN_FEATURES_EXCEED={MIN_FEATURES_EXCEED}")
print(f"  Per-feature P99: vetor ({len(thresholds_pf_p99)},) -- deteccao via MIN_FEATURES_EXCEED={MIN_FEATURES_EXCEED}")
print(f"  Best-F1 (score): {threshold_best_f1_score:.4f} -- aplicado no anomaly score (max-ratio)")


# ## 10. Evaluation
# 
# Metrics computed on test set (unbiased) and validation set:
# 
# - Per-feature P95/P99 thresholds
# - Best-F1 threshold (optimized via PR curve on validation)
# - AUC-ROC and AUC-PR (threshold-independent)

# In[ ]:


def evaluate_perfeature(pf_errors, y_true, pf_thresholds, threshold_name,
                        min_features_exceed=MIN_FEATURES_EXCEED):
    """Avalia metricas usando thresholds per-feature."""
    n_exceeded = (pf_errors > pf_thresholds).sum(axis=1)
    y_pred = (n_exceeded >= min_features_exceed).astype(int)

    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    print(f"\n{'=' * 65}")
    print(f"  Threshold: {threshold_name} (per-feature, MIN_FEATURES_EXCEED={min_features_exceed})")
    print(f"{'=' * 65}")
    print(f"  Acuracia:   {acc:.4f}")
    print(f"  Precisao:   {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"  Predicoes positivas: {y_pred.sum()} / {len(y_pred)}")
    print(f"  VP: {((y_pred == 1) & (y_true == 1)).sum()}  |  "
          f"FP: {((y_pred == 1) & (y_true == 0)).sum()}  |  "
          f"FN: {((y_pred == 0) & (y_true == 1)).sum()}  |  "
          f"VN: {((y_pred == 0) & (y_true == 0)).sum()}")
    print(f"  Media features excedidas (anomalias): "
          f"{n_exceeded[y_true == 1].mean():.1f} / {pf_errors.shape[1]}")

    return {'threshold': threshold_name,
            'f1': float(f1), 'accuracy': float(acc),
            'precision': float(prec), 'recall': float(rec)}


def evaluate_score(scores, y_true, threshold, threshold_name):
    """Avalia metricas usando threshold no anomaly score (max-ratio)."""
    y_pred = (scores > threshold).astype(int)

    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    print(f"\n{'=' * 65}")
    print(f"  Threshold: {threshold_name} = {threshold:.4f} (anomaly score)")
    print(f"{'=' * 65}")
    print(f"  Acuracia:   {acc:.4f}")
    print(f"  Precisao:   {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"  Predicoes positivas: {y_pred.sum()} / {len(y_pred)}")
    print(f"  VP: {((y_pred == 1) & (y_true == 1)).sum()}  |  "
          f"FP: {((y_pred == 1) & (y_true == 0)).sum()}  |  "
          f"FN: {((y_pred == 0) & (y_true == 1)).sum()}  |  "
          f"VN: {((y_pred == 0) & (y_true == 0)).sum()}")

    return {'threshold': threshold_name, 'value': float(threshold),
            'f1': float(f1), 'accuracy': float(acc),
            'precision': float(prec), 'recall': float(rec)}


auc_roc_test = roc_auc_score(y_test, scores_test_all)
auc_pr_test  = average_precision_score(y_test, scores_test_all)
auc_roc_val  = roc_auc_score(y_val, scores_val_all)
auc_pr_val   = average_precision_score(y_val, scores_val_all)

print("METRICAS INDEPENDENTES DE THRESHOLD (anomaly score = max-ratio)")
print("=" * 65)
print(f"  Teste     - AUC-ROC: {auc_roc_test:.4f} | AUC-PR: {auc_pr_test:.4f}")
print(f"  Validacao - AUC-ROC: {auc_roc_val:.4f} | AUC-PR: {auc_pr_val:.4f}")

print("\n\nAVALIACAO NO CONJUNTO DE TESTE")
print("=" * 65)
metrics_test_p95 = evaluate_perfeature(pf_errors_test_all, y_test,
                                       thresholds_pf_p95, 'Per-feature P95')
metrics_test_p99 = evaluate_perfeature(pf_errors_test_all, y_test,
                                       thresholds_pf_p99, 'Per-feature P99')
metrics_test_bf1 = evaluate_score(scores_test_all, y_test,
                                  threshold_best_f1_score, 'Best-F1 (score)')

print("\n\nAVALIACAO NO CONJUNTO DE VALIDACAO")
print("=" * 65)
metrics_val_p95 = evaluate_perfeature(pf_errors_val_all, y_val,
                                      thresholds_pf_p95, 'Per-feature P95')
metrics_val_p99 = evaluate_perfeature(pf_errors_val_all, y_val,
                                      thresholds_pf_p99, 'Per-feature P99')
metrics_val_bf1 = evaluate_score(scores_val_all, y_val,
                                 threshold_best_f1_score, 'Best-F1 (score)')


# In[ ]:


# --- Exemplo de reconstrucao: janela normal vs anomalia ---
model.eval()

# Selecionar uma janela normal e uma anomala do teste
idx_normal = np.where(y_test == 0)[0]
idx_anomaly = np.where(y_test == 1)[0]

if len(idx_normal) > 0 and len(idx_anomaly) > 0:
    # Pegar janelas representativas (percentil 50 do anomaly score)
    median_norm_idx = idx_normal[np.argsort(scores_test_all[idx_normal])[len(idx_normal) // 2]]
    median_anom_idx = idx_anomaly[np.argsort(scores_test_all[idx_anomaly])[len(idx_anomaly) // 2]]

    n_feat_show = min(6, len(selected_features))

    fig, axes = plt.subplots(2, n_feat_show, figsize=(4 * n_feat_show, 8), sharey=False)

    for row, (win_idx, title_prefix) in enumerate(
            [(median_norm_idx, 'Normal'), (median_anom_idx, 'Anomalia')]):
        x_orig = ds_test_all[win_idx].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            x_recon = model(x_orig)
        orig = x_orig.cpu().numpy()[0]
        recon = x_recon.cpu().numpy()[0]

        for col in range(n_feat_show):
            ax = axes[row, col]
            ax.plot(orig[:, col], label='Original', linewidth=1.5)
            ax.plot(recon[:, col], label='Reconstruido', linewidth=1.5, linestyle='--')
            ax.set_title(f'{title_prefix}\n{selected_features[col][:20]}', fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if row == 1:
                ax.set_xlabel('Timestep')
            if col == 0:
                ax.set_ylabel('Valor normalizado')
                ax.legend(fontsize=7)

    plt.suptitle('Reconstrucao: Original vs Autoencoder (Teste)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("Sem janelas suficientes para exemplo de reconstrucao.")


# In[ ]:


# === DIAGNOSTICO: Qualidade da Reconstrucao do Autoencoder ===
#
# Verifica se o modelo captura a dinamica temporal ou apenas a media de cada feature.
#
# Arquitetura atual (sem mean pooling):
#   Encoder: Conv1d(n_feat->64->128) -> BiLSTM(256->128) -> MHA(128) -> LayerNorm
#   Decoder: LSTM(128->128) -> LSTM(128->128) -> FC(256) -> FC(n_features)
#   A sequencia temporal completa e preservada ao longo do encoder e decoder.

model.eval()
sample = ds_test_all[0].unsqueeze(0).to(DEVICE)
with torch.no_grad():
    recon_sample = model(sample)

orig_np  = sample.cpu().numpy()[0]        # (WINDOW_SIZE, n_features)
recon_np = recon_sample.cpu().numpy()[0]  # (WINDOW_SIZE, n_features)

# 1) Variacao temporal: std ao longo dos WINDOW_SIZE timesteps, por feature
orig_std_per_feat  = orig_np.std(axis=0)
recon_std_per_feat = recon_np.std(axis=0)
ratio_global = recon_std_per_feat.mean() / (orig_std_per_feat.mean() + 1e-10)

print("=== DIAGNOSTICO DA RECONSTRUCAO ===\n")
print(f"Shape: original={orig_np.shape}, reconstruido={recon_np.shape}")
print(f"\nVariacao temporal (std ao longo de {WINDOW_SIZE} timesteps):")
print(f"  Original  - std media: {orig_std_per_feat.mean():.6f}, max: {orig_std_per_feat.max():.6f}")
print(f"  Recon     - std media: {recon_std_per_feat.mean():.6f}, max: {recon_std_per_feat.max():.6f}")
print(f"  Ratio recon/orig (std): {ratio_global:.4f}")
print()

# 2) Proporcao de features com reconstrucao 'flat' (colapso para media)
n_flat      = (recon_std_per_feat < 0.01).sum()
n_very_flat = (recon_std_per_feat < 0.001).sum()
print(f"Features com reconstrucao 'flat'       (std < 0.010): {n_flat}/{n_features} ({n_flat/n_features*100:.1f}%)")
print(f"Features com reconstrucao 'muito flat'  (std < 0.001): {n_very_flat}/{n_features} ({n_very_flat/n_features*100:.1f}%)")
print()

# 3) Range da reconstrucao (deve estar proximo de [0, 1] apos MinMax)
print(f"Range Original - min: {orig_np.min():.4f}, max: {orig_np.max():.4f}, mean: {orig_np.mean():.4f}")
print(f"Range Recon    - min: {recon_np.min():.4f}, max: {recon_np.max():.4f}, mean: {recon_np.mean():.4f}")
print()

# 4) Interpretacao automatica do ratio
if ratio_global < 0.3:
    print("[ALERTA] Reconstrucao muito flat (ratio < 0.3).")
    print("  O decoder pode estar colapsando para a media de cada feature.")
    print("  Sugestoes: reduzir n_features via feature selection mais agressiva,")
    print("  ou aumentar capacidade do decoder (LSTM maior / FC com mais camadas).")
elif ratio_global < 0.7:
    print("[ATENCAO] Reconstrucao parcialmente flat (0.3 <= ratio < 0.7).")
    print("  O modelo captura tendencia geral mas perde dinamica de alta frequencia.")
    print("  Avalie quais features tem maior MAE para identificar os mais discriminativos.")
else:
    print("[OK] Reconstrucao com boa variacao temporal (ratio >= 0.7).")

# 5) Top features melhor e pior reconstruidas
ratio_per_feat = recon_std_per_feat / (orig_std_per_feat + 1e-10)

print(f"\nTop 5 features com ratio mais proximo de 1.0 (melhor reconstrucao):")
best_idx = np.argsort(np.abs(ratio_per_feat - 1.0))[:5]
for i in best_idx:
    print(f"  {selected_features[i]:<40s}  orig_std={orig_std_per_feat[i]:.4f}  "
          f"recon_std={recon_std_per_feat[i]:.4f}  ratio={ratio_per_feat[i]:.3f}")

print(f"\nTop 5 features com reconstrucao mais flat (menor recon_std):")
flat_idx = np.argsort(recon_std_per_feat)[:5]
for i in flat_idx:
    print(f"  {selected_features[i]:<40s}  orig_std={orig_std_per_feat[i]:.4f}  "
          f"recon_std={recon_std_per_feat[i]:.6f}")


# ## 10.1 Per-Event Detection Rate
# 
# Detection rate per anomaly event: percentage of anomalous windows correctly classified.

# In[ ]:


anomaly_event_ids = set(event_info[event_info['event_label'] == 'anomaly']['event_id'].values)

def evaluate_per_event_pf(pf_errors, eids, y_true, pf_thresholds, threshold_name,
                          min_features_exceed=MIN_FEATURES_EXCEED):
    """Avalia taxa de deteccao por evento usando thresholds per-feature."""
    n_exceeded = (pf_errors > pf_thresholds).sum(axis=1)
    y_pred = (n_exceeded >= min_features_exceed).astype(int)

    results = []
    for eid in sorted(anomaly_event_ids):
        mask = (eids == eid)
        if mask.sum() == 0:
            continue
        event_labels = y_true[mask]
        event_preds = y_pred[mask]
        anom_mask = event_labels == 1
        if anom_mask.sum() == 0:
            continue
        n_anom = int(anom_mask.sum())
        detected = int(event_preds[anom_mask].sum())
        detection_rate = detected / n_anom
        results.append({
            'event_id': int(eid),
            'n_anom_windows': n_anom,
            'detected': detected,
            'detection_rate': detection_rate
        })

    if not results:
        print(f"  Nenhum evento de anomalia encontrado no split.")
        return results

    df_res = pd.DataFrame(results)
    print(f"\nAvaliacao por evento -- {threshold_name} (MIN_FEATURES_EXCEED={min_features_exceed})")
    print(f"{'=' * 65}")
    print(f"{'Evento':>8s} | {'Janelas Anom':>13s} | {'Detectadas':>11s} | {'Taxa Deteccao':>14s}")
    print(f"{'-' * 65}")
    for _, r in df_res.iterrows():
        print(f"{int(r['event_id']):>8d} | {int(r['n_anom_windows']):>13d} | "
              f"{int(r['detected']):>11d} | {r['detection_rate']:>13.1%}")
    print(f"{'-' * 65}")
    avg_rate = df_res['detection_rate'].mean()
    weighted_rate = df_res['detected'].sum() / df_res['n_anom_windows'].sum()
    print(f"  Media (por evento):     {avg_rate:.1%}")
    print(f"  Media (ponderada):      {weighted_rate:.1%}")
    print(f"  Eventos com deteccao>0: {(df_res['detection_rate'] > 0).sum()}/{len(df_res)}")
    return results


print("AVALIACAO POR EVENTO -- CONJUNTO DE TESTE")
event_results_p95 = evaluate_per_event_pf(pf_errors_test_all, eid_te, y_test,
                                          thresholds_pf_p95, 'Per-feature P95')
event_results_bf1 = evaluate_per_event_pf(pf_errors_test_all, eid_te, y_test,
                                          thresholds_pf_p99, 'Per-feature P99')


# In[ ]:


# --- Confusion Matrix (Best-F1 threshold) ---
y_pred_bf1 = (scores_test_all > threshold_best_f1_score).astype(int)
cm = confusion_matrix(y_test, y_pred_bf1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap com valores absolutos
ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Anomalia']).plot(
    ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title('Matriz de Confusao -- Teste (Best-F1)', fontsize=13, fontweight='bold')

# Heatmap normalizado por linha (%)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', ax=axes[1],
            xticklabels=['Normal', 'Anomalia'], yticklabels=['Normal', 'Anomalia'])
axes[1].set_xlabel('Predicao')
axes[1].set_ylabel('Real')
axes[1].set_title('Matriz de Confusao Normalizada (%)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

# --- Per-event detection bar chart ---
if event_results_p95:
    df_events = pd.DataFrame(event_results_p95)
    df_events = df_events.sort_values('detection_rate', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(df_events) * 0.5)))
    colors_ev = ['#C44E52' if r < 0.5 else '#55A868' for r in df_events['detection_rate']]
    bars = ax.barh(df_events['event_id'].astype(str), df_events['detection_rate'] * 100,
                   color=colors_ev, edgecolor='black', linewidth=0.5)
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax.set_xlabel('Taxa de Deteccao (%)')
    ax.set_ylabel('Event ID')
    ax.set_title('Taxa de Deteccao por Evento -- Per-feature P95 (Teste)', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.3)
    for bar, rate in zip(bars, df_events['detection_rate']):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{rate*100:.0f}%', va='center', fontsize=9)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
else:
    print("Nenhum evento de anomalia para plotar.")


# ## 10. Analise de Features Informativas (v4)
# Calcula:
# - feature_info_mask: features com std do erro > 0.01 nas janelas saudaveis de val
# - feature_auroc: AUROC por feature usando pf_errors_val_all vs y_val (usa labels)

print("\n=== SECAO 10: ANALISE DE FEATURES INFORMATIVAS ===\n")

# Mascara de features informativas: std do erro > 0.01 nas janelas saudaveis de val
feature_err_std_val_h = pf_errors_val_h.std(axis=0)
feature_info_mask = feature_err_std_val_h > 0.01
n_informative = feature_info_mask.sum()
print(f"Features informativas (std erro > 0.01): {n_informative}/{n_features} ({n_informative/n_features*100:.1f}%)")
print(f"Features planas (std erro <= 0.01): {n_features - n_informative}/{n_features} ({(n_features-n_informative)/n_features*100:.1f}%)")

# AUROC por feature (usa labels de validacao -- semi-supervisionado)
print("\nCalculando AUROC por feature (validacao)...")
t0 = time.time()
feature_auroc = np.zeros(n_features)
for i in range(n_features):
    try:
        feature_auroc[i] = roc_auc_score(y_val, pf_errors_val_all[:, i])
    except Exception:
        feature_auroc[i] = 0.5
print(f"AUROC calculado em {time.time()-t0:.1f}s")

discriminative_mask = feature_auroc > 0.55
n_discriminative = discriminative_mask.sum()
print(f"Features discriminativas (AUROC > 0.55): {n_discriminative}/{n_features} ({n_discriminative/n_features*100:.1f}%)")
print(f"AUROC medio (todas): {feature_auroc.mean():.4f}")
print(f"AUROC medio (informativas): {feature_auroc[feature_info_mask].mean():.4f}")
print(f"AUROC top 10:")
top10_idx = np.argsort(feature_auroc)[::-1][:10]
for i in top10_idx:
    print(f"  [{i:3d}] {selected_features[i]:<45s} AUROC={feature_auroc[i]:.4f}  std_err={feature_err_std_val_h[i]:.5f}")

# Distribuicao de AUROC
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].hist(feature_auroc, bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(x=0.55, color='red', linestyle='--', linewidth=2, label='Limiar AUROC=0.55')
axes[0].set_xlabel('AUROC por feature')
axes[0].set_ylabel('Contagem')
axes[0].set_title('Distribuicao de AUROC por Feature (Validacao)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(feature_err_std_val_h, feature_auroc,
                c=discriminative_mask.astype(int), cmap='RdYlGn', alpha=0.5, s=10)
axes[1].axvline(x=0.01, color='blue', linestyle='--', linewidth=1.5, label='std=0.01')
axes[1].axhline(y=0.55, color='red', linestyle='--', linewidth=1.5, label='AUROC=0.55')
axes[1].set_xlabel('Std do erro (janelas saudaveis val)')
axes[1].set_ylabel('AUROC')
axes[1].set_title('Std do Erro vs AUROC por Feature')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR_V4, 'feature_analysis.png'), dpi=120, bbox_inches='tight')
plt.show()

# Salvar mascaras
np.savez(
    os.path.join(OUTPUT_DIR_V4, 'feature_masks.npz'),
    info_mask=feature_info_mask,
    discriminative_mask=discriminative_mask,
    feature_auroc=feature_auroc,
    feature_err_std_val_h=feature_err_std_val_h,
)
print(f"\nMascaras salvas: {OUTPUT_DIR_V4}/feature_masks.npz")


# ## 11. Score Ponderado + Threshold Beta-F1 (v4)
# Melhoria 11.1: score ponderado por AUROC das features
# Melhoria 11.2: threshold otimizado por F-beta (beta=0.5) - prioriza precisao

print("\n=== SECAO 11: SCORE PONDERADO + BETA-F1 ===\n")


def compute_weighted_score(pf_errors, thresholds_p95, weights):
    """Score ponderado: soma ponderada dos ratios erro/threshold.
    weights: vetor de pesos por feature (ex: feature_auroc)
    """
    ratios = pf_errors / (thresholds_p95 + 1e-8)
    w = np.where(weights > 0, weights, 1e-10)
    return (ratios * w).sum(axis=1) / w.sum()


def find_threshold_beta_f1(scores, y_true, beta=0.5):
    """Otimiza threshold via F-beta. beta<1 prioriza precisao, beta>1 prioriza recall."""
    prec, rec, thrs = precision_recall_curve(y_true, scores)
    f_beta = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-8)
    if len(thrs) == 0:
        raise RuntimeError("precision_recall_curve retornou thresholds vazio.")
    idx = int(np.argmax(f_beta[:-1]))
    return float(thrs[idx]), float(prec[idx]), float(rec[idx]), float(f_beta[idx])


# Pesos = AUROC por feature (features planas/aleatorias tem peso ~0.5, discriminativas tem peso alto)
# Centramos em 0 em relacao a 0.5 (aleatorio) para que features aleatorias contribuam pouco
feature_weights = np.maximum(feature_auroc - 0.5, 0)  # Apenas contribuicao acima do aleatorio
print(f"Pesos por AUROC (> 0.5): nao-zero={(feature_weights > 0).sum()}/{n_features}, "
      f"max={feature_weights.max():.4f}, media={feature_weights[feature_weights>0].mean():.4f}")

# Computar scores ponderados
weighted_scores_val  = compute_weighted_score(pf_errors_val_all, thresholds_pf_p95, feature_weights)
weighted_scores_test = compute_weighted_score(pf_errors_test_all, thresholds_pf_p95, feature_weights)

print(f"\nScore ponderado (max-ratio pesado por AUROC):")
print(f"  Validacao - media: {weighted_scores_val.mean():.4f}, max: {weighted_scores_val.max():.4f}")
print(f"  Teste     - media: {weighted_scores_test.mean():.4f}, max: {weighted_scores_test.max():.4f}")

# AUC com score ponderado
auc_roc_weighted_val  = roc_auc_score(y_val, weighted_scores_val)
auc_roc_weighted_test = roc_auc_score(y_test, weighted_scores_test)
auc_pr_weighted_val   = average_precision_score(y_val, weighted_scores_val)
auc_pr_weighted_test  = average_precision_score(y_test, weighted_scores_test)
print(f"\n  AUC-ROC: val={auc_roc_weighted_val:.4f}, teste={auc_roc_weighted_test:.4f}")
print(f"  AUC-PR:  val={auc_pr_weighted_val:.4f}, teste={auc_pr_weighted_test:.4f}")
print(f"  AUC-ROC v3 baseline (referencia): val={auc_roc_val:.4f}, teste={auc_roc_test:.4f}")

# Threshold Beta-F1 no score ponderado (val)
for beta in [1.0, 0.5, 0.25]:
    thr, prec_v, rec_v, fb_v = find_threshold_beta_f1(weighted_scores_val, y_val, beta=beta)
    y_pred_te = (weighted_scores_test > thr).astype(int)
    prec_te = precision_score(y_test, y_pred_te, zero_division=0)
    rec_te  = recall_score(y_test, y_pred_te, zero_division=0)
    f1_te   = f1_score(y_test, y_pred_te, zero_division=0)
    print(f"\n  Beta-F1 (beta={beta:.2f}): threshold={thr:.4f}")
    print(f"    Val:   prec={prec_v:.4f}, rec={rec_v:.4f}")
    print(f"    Teste: prec={prec_te:.4f}, rec={rec_te:.4f}, F1={f1_te:.4f}")
    print(f"    Positivos: {y_pred_te.sum():,}/{len(y_pred_te):,} ({y_pred_te.mean()*100:.1f}%)")

# Selecionar beta=0.5 para analise detalhada
beta_target = 0.5
thr_weighted_beta05, _, _, _ = find_threshold_beta_f1(weighted_scores_val, y_val, beta=beta_target)
y_pred_weighted_val  = (weighted_scores_val > thr_weighted_beta05).astype(int)
y_pred_weighted_test = (weighted_scores_test > thr_weighted_beta05).astype(int)

print(f"\n=== Avaliacao detalhada: Score Ponderado + Beta-F1 (beta={beta_target}) ===")
for name, y_true, y_pred in [('Validacao', y_val, y_pred_weighted_val),
                               ('Teste', y_test, y_pred_weighted_test)]:
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    acc  = accuracy_score(y_true, y_pred)
    vp = int(((y_pred==1)&(y_true==1)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    vn = int(((y_pred==0)&(y_true==0)).sum())
    print(f"\n  {name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    print(f"    VP={vp}, FP={fp}, FN={fn}, VN={vn}")
    print(f"    Positivos: {y_pred.sum():,}/{len(y_pred):,} ({y_pred.mean()*100:.1f}%)")


# ## 12. Suavizacao Temporal por Majority Vote (v4)
# Janelas consecutivas do mesmo evento com voto majoritario.
# Eventos reais duram centenas de janelas; picos isolados sao FP.

print("\n=== SECAO 12: SUAVIZACAO TEMPORAL ===\n")


def apply_temporal_smoothing(y_pred, eids, K=5):
    """Majority vote sobre K janelas consecutivas, por evento.
    Uma janela e reclassificada como anomalia apenas se >= ceil(K/2) vizinhos tambem o forem.
    Processado por evento (nao cruza limites de evento).
    """
    y_smooth = y_pred.copy().astype(float)
    for eid in np.unique(eids):
        mask = (eids == eid)
        s = pd.Series(y_pred[mask].astype(float))
        smoothed = (s.rolling(K, center=True, min_periods=1).mean() >= 0.5).values.astype(float)
        y_smooth[mask] = smoothed
    return y_smooth.astype(int)


# Testar suavizacao no score ponderado + beta-F1 (melhor estrategia Tier 1)
print("Impacto da suavizacao temporal no Score Ponderado + Beta-F1 (beta=0.5):")
print(f"{'K':>5s} | {'Precisao':>10s} | {'Recall':>8s} | {'F1':>8s} | {'Pos (%)':>10s}")
print("-" * 50)

best_k_f1 = -1
best_k_val = 1
for K in [1, 3, 5, 9, 15]:
    y_smooth_te = apply_temporal_smoothing(y_pred_weighted_test, eid_te, K=K)
    prec = precision_score(y_test, y_smooth_te, zero_division=0)
    rec  = recall_score(y_test, y_smooth_te, zero_division=0)
    f1   = f1_score(y_test, y_smooth_te, zero_division=0)
    pos_pct = y_smooth_te.mean() * 100
    label = " <- baseline" if K == 1 else ""
    print(f"  K={K:>2d} | {prec:>10.4f} | {rec:>8.4f} | {f1:>8.4f} | {pos_pct:>9.1f}%{label}")
    if f1 > best_k_f1:
        best_k_f1 = f1
        best_k_val = K

print(f"\nMelhor K (por F1 no teste): K={best_k_val}")

# Aplicar suavizacao com K=5 no val e test
y_smooth_val_k5  = apply_temporal_smoothing(y_pred_weighted_val, eid_va, K=5)
y_smooth_test_k5 = apply_temporal_smoothing(y_pred_weighted_test, eid_te, K=5)

print("\n=== Avaliacao detalhada: Score Ponderado + Beta-F1 + Suavizacao (K=5) ===")
for name, y_true, y_pred in [('Validacao', y_val, y_smooth_val_k5),
                               ('Teste', y_test, y_smooth_test_k5)]:
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    acc  = accuracy_score(y_true, y_pred)
    vp = int(((y_pred==1)&(y_true==1)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    vn = int(((y_pred==0)&(y_true==0)).sum())
    print(f"\n  {name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    print(f"    VP={vp}, FP={fp}, FN={fn}, VN={vn}")


# ## 13. Regra de Contagem com Mascara Informativa (v4)
# Exclui features planas da contagem de features excedidas.

print("\n=== SECAO 13: CONTAGEM COM MASCARA INFORMATIVA ===\n")


def evaluate_perfeature_v4(pf_errors, y_true, pf_thresholds, info_mask, label,
                            min_frac=0.05):
    """Avalia thresholds per-feature usando apenas features informativas."""
    errs = pf_errors[:, info_mask]
    thrs = pf_thresholds[info_mask]
    n_informative_local = int(info_mask.sum())
    min_exceeded = max(1, int(n_informative_local * min_frac))
    n_exceeded = (errs > thrs).sum(axis=1)
    y_pred = (n_exceeded >= min_exceeded).astype(int)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    acc  = accuracy_score(y_true, y_pred)

    print(f"\n  {label} (n_informative={n_informative_local}, min_exceeded={min_exceeded}):")
    print(f"    Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    print(f"    Positivos: {y_pred.sum():,}/{len(y_pred):,} ({y_pred.mean()*100:.1f}%)")
    vp = int(((y_pred==1)&(y_true==1)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    vn = int(((y_pred==0)&(y_true==0)).sum())
    print(f"    VP={vp}, FP={fp}, FN={fn}, VN={vn}")
    return y_pred, {'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc}


# Testar diferentes mascaras e min_frac
print("Comparando mascaras de features (Teste):")
print(f"\n  [v3 baseline] Todas as {n_features} features, min_features_exceed={MIN_FEATURES_EXCEED}:")
y_v3_p95 = ((pf_errors_test_all > thresholds_pf_p95).sum(axis=1) >= MIN_FEATURES_EXCEED).astype(int)
prec_v3 = precision_score(y_test, y_v3_p95, zero_division=0)
rec_v3  = recall_score(y_test, y_v3_p95, zero_division=0)
f1_v3   = f1_score(y_test, y_v3_p95, zero_division=0)
print(f"    Prec={prec_v3:.4f}, Rec={rec_v3:.4f}, F1={f1_v3:.4f}")

y_info_p95, metrics_info_p95 = evaluate_perfeature_v4(
    pf_errors_test_all, y_test, thresholds_pf_p95, feature_info_mask,
    "P95 + Mascara Info (std>0.01)", min_frac=0.05)

y_disc_p95, metrics_disc_p95 = evaluate_perfeature_v4(
    pf_errors_test_all, y_test, thresholds_pf_p95, discriminative_mask,
    "P95 + Mascara Discriminativa (AUROC>0.55)", min_frac=0.05)

y_info_p99, metrics_info_p99 = evaluate_perfeature_v4(
    pf_errors_test_all, y_test, thresholds_pf_p99, feature_info_mask,
    "P99 + Mascara Info", min_frac=0.05)

# Testar suavizacao em cima das mascaras
print("\n  P95 + Mascara Info + Suavizacao K=5 (Teste):")
y_info_smooth = apply_temporal_smoothing(y_info_p95, eid_te, K=5)
prec = precision_score(y_test, y_info_smooth, zero_division=0)
rec  = recall_score(y_test, y_info_smooth, zero_division=0)
f1   = f1_score(y_test, y_info_smooth, zero_division=0)
print(f"    Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")


# ## 14. Classificador XGBoost Pos-Hoc (v4)
# Treina XGBoost nos vetores de erro de reconstrucao do set de validacao
# com labels de anomalia disponiveis. Avalia no teste (sem vazamento).

print("\n=== SECAO 14: CLASSIFICADOR XGBOOST POS-HOC ===\n")

try:
    import xgboost as xgb
    print(f"XGBoost version: {xgb.__version__}")
    XGBOOST_AVAILABLE = True
except ImportError:
    print("AVISO: XGBoost nao instalado. Pulando Secao 14.")
    print("  Instale com: pip install xgboost")
    XGBOOST_AVAILABLE = False


def build_classifier_features(pf_errors, thresholds_p95, feature_weights):
    """Constroi matriz de features para o classificador XGBoost.
    Combina erros brutos, ratios, flags, estatisticas globais e scores derivados.
    """
    ratios = pf_errors / (thresholds_p95 + 1e-8)
    exceeded = (pf_errors > thresholds_p95).astype(np.float32)

    # Score max-ratio (v3 baseline)
    score_maxratio = ratios.max(axis=1, keepdims=True)

    # Score ponderado (v4)
    w = np.where(feature_weights > 0, feature_weights, 1e-10)
    score_weighted = ((ratios * w).sum(axis=1) / w.sum()).reshape(-1, 1)

    # Contagem features excedidas (P95)
    n_exceeded = exceeded.sum(axis=1, keepdims=True)

    # Estatisticas globais do erro
    err_mean = pf_errors.mean(axis=1, keepdims=True)
    err_std  = pf_errors.std(axis=1, keepdims=True)
    err_max  = pf_errors.max(axis=1, keepdims=True)
    err_p95  = np.percentile(pf_errors, 95, axis=1).reshape(-1, 1)
    err_p99  = np.percentile(pf_errors, 99, axis=1).reshape(-1, 1)

    # Estatisticas dos ratios
    ratio_mean = ratios.mean(axis=1, keepdims=True)
    ratio_std  = ratios.std(axis=1, keepdims=True)

    X = np.concatenate([
        pf_errors,          # n_features: erros brutos
        ratios,             # n_features: ratios erro/threshold
        exceeded,           # n_features: flags binarios
        score_maxratio,     # 1: score v3
        score_weighted,     # 1: score ponderado v4
        n_exceeded,         # 1: contagem features excedidas
        err_mean,           # 1
        err_std,            # 1
        err_max,            # 1
        err_p95,            # 1
        err_p99,            # 1
        ratio_mean,         # 1
        ratio_std,          # 1
    ], axis=1).astype(np.float32)

    return X


if XGBOOST_AVAILABLE:
    print("Construindo features para XGBoost...")
    X_val  = build_classifier_features(pf_errors_val_all, thresholds_pf_p95, feature_weights)
    X_test = build_classifier_features(pf_errors_test_all, thresholds_pf_p95, feature_weights)
    print(f"  X_val:  {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  Anomalias em val: {y_val.sum():,}/{len(y_val):,} ({y_val.mean()*100:.2f}%)")
    print(f"  Anomalias em test: {y_test.sum():,}/{len(y_test):,} ({y_test.mean()*100:.2f}%)")

    scale_pos_weight = int((y_val == 0).sum() / (y_val == 1).sum())
    print(f"  scale_pos_weight: {scale_pos_weight}")

    print("\nTreinando XGBoost no set de validacao (com labels)...")
    t0 = time.time()

    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='aucpr',
        random_state=SEED,
        tree_method='hist',
        device='cuda' if DEVICE.type == 'cuda' else 'cpu',
        n_jobs=-1,
    )
    clf.fit(X_val, y_val, verbose=False)
    print(f"XGBoost treinado em {time.time()-t0:.1f}s")

    # Probabilidades no val (para selecionar threshold) e no test (avaliacao final)
    prob_val  = clf.predict_proba(X_val)[:, 1]
    prob_test = clf.predict_proba(X_test)[:, 1]

    auc_roc_xgb_val  = roc_auc_score(y_val, prob_val)
    auc_roc_xgb_test = roc_auc_score(y_test, prob_test)
    auc_pr_xgb_val   = average_precision_score(y_val, prob_val)
    auc_pr_xgb_test  = average_precision_score(y_test, prob_test)
    print(f"\nXGBoost AUC-ROC: val={auc_roc_xgb_val:.4f}, teste={auc_roc_xgb_test:.4f}")
    print(f"XGBoost AUC-PR:  val={auc_pr_xgb_val:.4f}, teste={auc_pr_xgb_test:.4f}")

    # Selecionar threshold por Beta-F1 (beta=0.5) no val
    print("\nSelecionando threshold Beta-F1 (beta=0.5) no val...")
    thr_xgb, prec_xgb_val, rec_xgb_val, fbeta_xgb_val = find_threshold_beta_f1(
        prob_val, y_val, beta=0.5)
    print(f"  Threshold XGBoost (beta=0.5): {thr_xgb:.4f}")
    print(f"  Val: prec={prec_xgb_val:.4f}, rec={rec_xgb_val:.4f}")

    # Avaliar no teste
    y_pred_xgb_test = (prob_test > thr_xgb).astype(int)
    prec_xgb_test = precision_score(y_test, y_pred_xgb_test, zero_division=0)
    rec_xgb_test  = recall_score(y_test, y_pred_xgb_test, zero_division=0)
    f1_xgb_test   = f1_score(y_test, y_pred_xgb_test, zero_division=0)
    acc_xgb_test  = accuracy_score(y_test, y_pred_xgb_test)
    vp_xgb = int(((y_pred_xgb_test==1)&(y_test==1)).sum())
    fp_xgb = int(((y_pred_xgb_test==1)&(y_test==0)).sum())
    fn_xgb = int(((y_pred_xgb_test==0)&(y_test==1)).sum())
    vn_xgb = int(((y_pred_xgb_test==0)&(y_test==0)).sum())

    print(f"\n=== Avaliacao detalhada: XGBoost (beta=0.5, Teste) ===")
    print(f"  Acc={acc_xgb_test:.4f}, Prec={prec_xgb_test:.4f}, Rec={rec_xgb_test:.4f}, F1={f1_xgb_test:.4f}")
    print(f"  VP={vp_xgb}, FP={fp_xgb}, FN={fn_xgb}, VN={vn_xgb}")
    print(f"  Positivos: {y_pred_xgb_test.sum():,}/{len(y_pred_xgb_test):,} ({y_pred_xgb_test.mean()*100:.1f}%)")

    # XGBoost + Suavizacao temporal K=5
    y_pred_xgb_smooth_test = apply_temporal_smoothing(y_pred_xgb_test, eid_te, K=5)
    prec_xs = precision_score(y_test, y_pred_xgb_smooth_test, zero_division=0)
    rec_xs  = recall_score(y_test, y_pred_xgb_smooth_test, zero_division=0)
    f1_xs   = f1_score(y_test, y_pred_xgb_smooth_test, zero_division=0)
    print(f"\n  XGBoost + Suavizacao K=5 (Teste):")
    print(f"  Prec={prec_xs:.4f}, Rec={rec_xs:.4f}, F1={f1_xs:.4f}")
    print(f"  Positivos: {y_pred_xgb_smooth_test.sum():,}/{len(y_pred_xgb_smooth_test):,}")

    # Feature importance (top 30)
    fi = clf.feature_importances_
    n_base_feats = n_features  # primeiras n_features colunas sao os erros brutos
    fi_per_original_feat = fi[:n_base_feats]
    top30 = np.argsort(fi_per_original_feat)[::-1][:30]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(30), fi_per_original_feat[top30][::-1],
            color='#4C72B0', edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(30))
    ax.set_yticklabels([selected_features[i] for i in top30][::-1], fontsize=8)
    ax.set_xlabel('Feature Importance (XGBoost, erros brutos)')
    ax.set_title('Top 30 Features por Importancia no XGBoost (erros de reconstrucao)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_V4, 'xgb_feature_importance.png'), dpi=120, bbox_inches='tight')
    plt.show()

    # Curvas ROC e PR
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, prob_test)
    axes[0].plot(fpr_xgb, tpr_xgb, 'b-', linewidth=2,
                 label=f'XGBoost v4 (AUC={auc_roc_xgb_test:.4f})')
    fpr_v3, tpr_v3, _ = roc_curve(y_test, scores_test_all)
    axes[0].plot(fpr_v3, tpr_v3, 'r--', linewidth=1.5, alpha=0.7,
                 label=f'Max-Ratio v3 (AUC={auc_roc_test:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4)
    axes[0].set_xlabel('FPR')
    axes[0].set_ylabel('TPR')
    axes[0].set_title('Curva ROC -- Teste')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    prec_xgb_curve, rec_xgb_curve, _ = precision_recall_curve(y_test, prob_test)
    axes[1].plot(rec_xgb_curve, prec_xgb_curve, 'b-', linewidth=2,
                 label=f'XGBoost v4 (AUC-PR={auc_pr_xgb_test:.4f})')
    prec_v3_curve, rec_v3_curve, _ = precision_recall_curve(y_test, scores_test_all)
    axes[1].plot(rec_v3_curve, prec_v3_curve, 'r--', linewidth=1.5, alpha=0.7,
                 label=f'Max-Ratio v3 (AUC-PR={auc_pr_test:.4f})')
    baseline_pr = y_test.sum() / len(y_test)
    axes[1].axhline(y=baseline_pr, color='k', linestyle=':', alpha=0.5,
                    label=f'Prevalencia={baseline_pr:.4f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precisao')
    axes[1].set_title('Curva Precision-Recall -- Teste')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_V4, 'roc_pr_curves.png'), dpi=120, bbox_inches='tight')
    plt.show()

    # Salvar XGBoost
    xgb_clf_path = os.path.join(OUTPUT_DIR_V4, 'xgb_classifier.pkl')
    joblib.dump(clf, xgb_clf_path)
    xgb_thr_path = os.path.join(OUTPUT_DIR_V4, 'xgb_threshold.json')
    with open(xgb_thr_path, 'w') as f_out:
        json.dump({
            'threshold': thr_xgb,
            'beta': 0.5,
            'auc_roc_val': float(auc_roc_xgb_val),
            'auc_roc_test': float(auc_roc_xgb_test),
            'precision_test': float(prec_xgb_test),
            'recall_test': float(rec_xgb_test),
            'f1_test': float(f1_xgb_test),
        }, f_out, indent=2)
    print(f"\nXGBoost salvo: {xgb_clf_path}")
    print(f"Threshold XGBoost salvo: {xgb_thr_path}")

else:
    # Valores dummy para tabela comparativa
    prec_xgb_test = rec_xgb_test = f1_xgb_test = float('nan')
    auc_roc_xgb_test = float('nan')
    y_pred_xgb_test = np.zeros(len(y_test), dtype=int)
    y_pred_xgb_smooth_test = np.zeros(len(y_test), dtype=int)
    prec_xs = rec_xs = f1_xs = float('nan')
    prob_test = None


# ## 15. Avaliacao por Evento — Todas as Estrategias (v4)

print("\n=== SECAO 15: AVALIACAO POR EVENTO ===\n")


def evaluate_per_event_all(y_pred, eids, y_true, label):
    """Taxa de deteccao por evento de anomalia."""
    results = []
    for eid in sorted(anomaly_event_ids):
        mask = (eids == eid)
        if mask.sum() == 0:
            continue
        event_labels = y_true[mask]
        event_preds = y_pred[mask]
        anom_mask = event_labels == 1
        if anom_mask.sum() == 0:
            continue
        n_anom = int(anom_mask.sum())
        detected = int(event_preds[anom_mask].sum())
        results.append({
            'event_id': int(eid),
            'n_anom_windows': n_anom,
            'detected': detected,
            'detection_rate': detected / n_anom
        })

    if not results:
        return results

    df_res = pd.DataFrame(results)
    print(f"\n  {label}:")
    print(f"  {'Evento':>8s} | {'Janelas Anom':>13s} | {'Detectadas':>10s} | {'Taxa':>8s}")
    for _, r in df_res.iterrows():
        print(f"  {int(r['event_id']):>8d} | {int(r['n_anom_windows']):>13d} | "
              f"{int(r['detected']):>10d} | {r['detection_rate']:>7.1%}")
    avg = df_res['detection_rate'].mean()
    weighted = df_res['detected'].sum() / df_res['n_anom_windows'].sum()
    print(f"  Media (por evento): {avg:.1%} | Ponderada: {weighted:.1%}")
    return results


print("Taxa de deteccao por evento (Teste):")
evaluate_per_event_all(y_v3_p95, eid_te, y_test, "v3 P95 baseline")
evaluate_per_event_all(y_pred_weighted_test, eid_te, y_test,
                       "v4 Score Ponderado + Beta-F1 (beta=0.5)")
evaluate_per_event_all(y_smooth_test_k5, eid_te, y_test,
                       "v4 Score Ponderado + Beta-F1 + Suavizacao K=5")
evaluate_per_event_all(y_info_p95, eid_te, y_test,
                       "v4 P95 + Mascara Informativa")
if XGBOOST_AVAILABLE:
    evaluate_per_event_all(y_pred_xgb_test, eid_te, y_test,
                           "v4 XGBoost (beta=0.5)")
    evaluate_per_event_all(y_pred_xgb_smooth_test, eid_te, y_test,
                           "v4 XGBoost + Suavizacao K=5")


# ## 16. Tabela Comparativa Final (v3 vs v4)

print("\n" + "=" * 90)
print("TABELA COMPARATIVA: v3 vs v4 -- CONJUNTO DE TESTE")
print("=" * 90)


def compute_metrics_row(y_true, y_pred, score=None):
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    acc  = accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, score) if score is not None else float('nan')
    return prec, rec, f1, acc, auc_roc


rows = []

# v3 baseline
y_v3_p99 = ((pf_errors_test_all > thresholds_pf_p99).sum(axis=1) >= MIN_FEATURES_EXCEED).astype(int)
y_v3_bf1 = (scores_test_all > threshold_best_f1_score).astype(int)

for label, y_pred, score in [
    ("v3 P95 baseline",      y_v3_p95,                  scores_test_all),
    ("v3 P99 baseline",      y_v3_p99,                  scores_test_all),
    ("v3 Best-F1 baseline",  y_v3_bf1,                  scores_test_all),
    ("v4 Score Pond.+BF1",   y_pred_weighted_test,      weighted_scores_test),
    ("v4 +Suav K=5",         y_smooth_test_k5,          None),
    ("v4 P95+MascInfo",      y_info_p95,                scores_test_all),
    ("v4 P95+MascInfo+K=5",  y_info_smooth,             None),
    ("v4 XGBoost",           y_pred_xgb_test,           prob_test if XGBOOST_AVAILABLE else None),
    ("v4 XGBoost+K=5",       y_pred_xgb_smooth_test,    None),
]:
    prec, rec, f1, acc, auc_roc_val = compute_metrics_row(y_test, y_pred, score)
    rows.append({'Estrategia': label, 'Precisao': prec, 'Recall': rec, 'F1': f1, 'AUC-ROC': auc_roc_val})

df_results = pd.DataFrame(rows)
print(df_results.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

print("\n" + "=" * 90)
print("RESUMO: Melhoria em Precisao (mantendo Recall >= 0.80)")
print("=" * 90)
baseline_prec = df_results.iloc[0]['Precisao']
for _, row in df_results.iterrows():
    if row['Recall'] >= 0.80:
        melhoria = (row['Precisao'] / baseline_prec - 1) * 100 if baseline_prec > 0 else 0
        flag = " <- MELHOR" if row['F1'] == df_results[df_results['Recall'] >= 0.80]['F1'].max() else ""
        print(f"  {row['Estrategia']:<30s} Prec={row['Precisao']:.4f} "
              f"Rec={row['Recall']:.4f} F1={row['F1']:.4f} "
              f"(+{melhoria:.0f}% vs baseline){flag}")


# ## 11. Visualizations

# In[ ]:


fig, axes = plt.subplots(2, 1, figsize=(16, 12))

ax = axes[0]
ax.scatter(range(len(scores_test_all)), scores_test_all, c=y_test, cmap='coolwarm',
           s=1, alpha=0.3, label='Score (azul=normal, vermelho=anomalia)')
ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=2,
           label='Limiar P95 (score=1.0)')
ax.axhline(y=threshold_best_f1_score, color='green', linestyle='--', linewidth=2,
           label=f'Best-F1 = {threshold_best_f1_score:.4f}')
ax.set_xlabel('Indice da Janela')
ax.set_ylabel('Anomaly Score (max-ratio)')
ax.set_title('Anomaly Score no Teste (max erro/threshold por feature)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(scores_test_all[y_test == 0], bins=200, alpha=0.6, label='Normal', density=True)
ax.hist(scores_test_all[y_test == 1], bins=200, alpha=0.6, label='Anomalia', density=True)
ax.axvline(x=1.0, color='orange', linestyle='--', linewidth=2,
           label='Limiar P95 (score=1.0)')
ax.axvline(x=threshold_best_f1_score, color='green', linestyle='--', linewidth=2,
           label=f'Best-F1 = {threshold_best_f1_score:.4f}')
ax.set_xlabel('Anomaly Score (max-ratio)')
ax.set_ylabel('Densidade')
ax.set_title('Distribuicao dos Anomaly Scores (Teste)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, np.percentile(scores_test_all, 99.5))

plt.tight_layout()
plt.show()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

fpr, tpr, _ = roc_curve(y_test, scores_test_all)
roc_auc_val_plot = auc(fpr, tpr)
ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc_val_plot:.4f})')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Aleatorio')
ax1.set_xlabel('Taxa de Falsos Positivos (FPR)')
ax1.set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
ax1.set_title('Curva ROC -- Conjunto de Teste (anomaly score)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1.05])

prec_test, rec_test, _ = precision_recall_curve(y_test, scores_test_all)
pr_auc_val_plot = auc(rec_test, prec_test)
ax2.plot(rec_test, prec_test, 'r-', linewidth=2,
         label=f'PR (AUC = {pr_auc_val_plot:.4f})')
baseline = y_test.sum() / len(y_test)
ax2.axhline(y=baseline, color='k', linestyle='--', alpha=0.5,
            label=f'Baseline (prevalencia = {baseline:.4f})')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precisao')
ax2.set_title('Curva Precision-Recall -- Conjunto de Teste (anomaly score)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1.05])

plt.tight_layout()
plt.show()


# In[ ]:


fig, axes = plt.subplots(2, 1, figsize=(16, 12))

ax = axes[0]
ax.scatter(range(len(scores_val_all)), scores_val_all, c=y_val, cmap='coolwarm',
           s=1, alpha=0.3, label='Score (azul=normal, vermelho=anomalia)')
ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=2,
           label='Limiar P95 (score=1.0)')
ax.axhline(y=threshold_best_f1_score, color='green', linestyle='--', linewidth=2,
           label=f'Best-F1 = {threshold_best_f1_score:.4f}')
ax.set_xlabel('Indice da Janela')
ax.set_ylabel('Anomaly Score (max-ratio)')
ax.set_title('Anomaly Score na Validacao (max erro/threshold por feature)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(scores_val_all[y_val == 0], bins=200, alpha=0.6, label='Normal', density=True)
ax.hist(scores_val_all[y_val == 1], bins=200, alpha=0.6, label='Anomalia', density=True)
ax.axvline(x=1.0, color='orange', linestyle='--', linewidth=2,
           label='Limiar P95 (score=1.0)')
ax.axvline(x=threshold_best_f1_score, color='green', linestyle='--', linewidth=2,
           label=f'Best-F1 = {threshold_best_f1_score:.4f}')
ax.set_xlabel('Anomaly Score (max-ratio)')
ax.set_ylabel('Densidade')
ax.set_title('Distribuicao dos Anomaly Scores (Validacao)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, np.percentile(scores_val_all, 99.5))

plt.tight_layout()
plt.show()


# ## 12. Artifacts

# In[ ]:


model_path = os.path.join(OUTPUT_DIR, 'modelo_cnn_bilstm_attention_autoencoder.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'n_features': n_features,
    'timesteps': WINDOW_SIZE,
    'architecture': 'CNNBiLSTMAttentionAutoencoder',
    'best_val_loss': best_val_loss,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'lr_history': lr_history,
    'dropout_rate': DROPOUT_RATE,
    'loss_function': 'MAE (L1Loss)',
    'bidirectional_encoder': True,
}, model_path)
print(f"Modelo salvo: {model_path}")

scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"Scaler salvo: {scaler_path}")

features_path = os.path.join(OUTPUT_DIR, 'features_selecionadas.json')
with open(features_path, 'w') as f:
    json.dump({
        'selected_features': selected_features,
        'n_original_features': len(sensor_cols),
        'n_selected_features': len(selected_features),
        'selection_method': 'Nao-supervisionada (variancia + correlacao com operacionais)',
        'variance_threshold': VARIANCE_THRESHOLD,
        'corr_with_operational_min': CORR_WITH_OPERATIONAL_MIN,
        'corr_inter_feature_max': CORR_INTER_FEATURE_MAX,
    }, f, indent=2, ensure_ascii=False)
print(f"Features salvas: {features_path}")

thresholds_pf_p95_path = os.path.join(OUTPUT_DIR, 'thresholds_per_feature_p95.npy')
np.save(thresholds_pf_p95_path, thresholds_pf_p95)
thresholds_pf_p99_path = os.path.join(OUTPUT_DIR, 'thresholds_per_feature_p99.npy')
np.save(thresholds_pf_p99_path, thresholds_pf_p99)
print(f"Thresholds per-feature salvos: P95 ({thresholds_pf_p95_path}), P99 ({thresholds_pf_p99_path})")

thresholds_path = os.path.join(OUTPUT_DIR, 'thresholds.json')
with open(thresholds_path, 'w') as f:
    json.dump({
        'threshold_type': 'per-feature',
        'threshold_best_f1_score': threshold_best_f1_score,
        'min_features_exceed': MIN_FEATURES_EXCEED,
        'computed_on': 'janelas saudaveis de validacao (P95/P99 per-feature) e curva PR (Best-F1 no anomaly score)',
        'note': 'Metricas de validacao sao otimistamente viesadas (mesmo split usado para ajuste de threshold). '
                'Usar metricas de TESTE como resultado final.',
        'fast_mode': FAST_MODE,
        'metrics_teste_p95': metrics_test_p95,
        'metrics_teste_p99': metrics_test_p99,
        'metrics_teste_best_f1': metrics_test_bf1,
        'auc_roc_test': auc_roc_test,
        'auc_pr_test': auc_pr_test,
        'auc_roc_val': auc_roc_val,
        'auc_pr_val': auc_pr_val,
    }, f, indent=2, ensure_ascii=False)
print(f"Thresholds metadata salvos: {thresholds_path}")

print(f"\nArtefatos salvos no diretorio: {os.path.abspath(OUTPUT_DIR)}/")
if FAST_MODE:
    print(f"FAST_MODE estava ativado -- resultados nao sao representativos do dataset completo.")

# --- Salvar artefatos v4 ---
np.savez(
    os.path.join(OUTPUT_DIR_V4, 'feature_masks.npz'),
    info_mask=feature_info_mask,
    discriminative_mask=discriminative_mask,
    feature_auroc=feature_auroc,
    feature_err_std_val_h=feature_err_std_val_h,
)
np.save(os.path.join(OUTPUT_DIR_V4, 'feature_weights.npy'), feature_weights)
np.save(os.path.join(OUTPUT_DIR_V4, 'feature_auroc.npy'), feature_auroc)

results_v4 = {
    'version': 'v4',
    'base_model': 'CNN-BiLSTM-Attention Autoencoder (treinado neste script)',
    'improvements': [
        'Feature-weighted anomaly score (by AUROC)',
        'Beta-F1 threshold (beta=0.5)',
        'Temporal smoothing K=5',
        'Informative feature mask (std>0.01)',
        'XGBoost post-hoc classifier',
    ],
    'n_features': n_features,
    'n_informative_features': int(n_informative),
    'n_discriminative_features': int(n_discriminative),
    'thresholds': {
        'weighted_score_beta05': float(thr_weighted_beta05),
        'xgboost_beta05': float(thr_xgb) if XGBOOST_AVAILABLE else None,
    },
    'metrics_test': {row['Estrategia']: {
        'precision': float(row['Precisao']),
        'recall': float(row['Recall']),
        'f1': float(row['F1']),
        'auc_roc': float(row['AUC-ROC']) if not np.isnan(row['AUC-ROC']) else None,
    } for _, row in df_results.iterrows()},
}

results_v4_path = os.path.join(OUTPUT_DIR_V4, 'results_v4.json')
with open(results_v4_path, 'w') as f:
    json.dump(results_v4, f, indent=2, ensure_ascii=False)

print(f"\nArtefatos v4 adicionais salvos em: {os.path.abspath(OUTPUT_DIR_V4)}/")
print(f"  - feature_masks.npz (info_mask, discriminative_mask, feature_auroc)")
print(f"  - feature_weights.npy, feature_auroc.npy")
print(f"  - results_v4.json (metricas de todas as estrategias)")
if XGBOOST_AVAILABLE:
    print(f"  - xgb_classifier.pkl, xgb_threshold.json")
    print(f"  - xgb_feature_importance.png, roc_pr_curves.png")
print(f"  - feature_analysis.png")

print("\n=== PIPELINE COMPLETO (v3 + v4) CONCLUIDO ===")


# ## Conclusao
#
# Pipeline completo de deteccao de anomalias semi-supervisionada para dados SCADA do Wind Farm C.
#
# **v3 — Baseline:**
# - Autoencoder CNN-BiLSTM-Attention treinado somente em dados saudaveis
# - Deteccao via erros de reconstrucao per-feature (thresholds P95/P99 + Best-F1)
#
# **v4 — Melhorias sem retreino:**
# - Score ponderado por AUROC por feature + threshold Beta-F1 (beta=0.5)
# - Suavizacao temporal por majority vote (K=5)
# - Mascara de features informativas (std erro > 0.01)
# - Classificador XGBoost pos-hoc nos vetores de erro de reconstrucao
#
# **Decisoes de design:**
# - Scaler ajustado somente em amostras saudaveis de treino
# - Selecao de features nao-supervisionada (variancia + correlacao operacional)
# - Metricas de validacao sao otimisticamente viesadas; metricas de TESTE sao o resultado final
#
# **Referencias:**
# - Ashkarkalaei et al. (2025). MSSP.
# - Chen et al. (2021). Renewable Energy.
# - Lee et al. (2024). Sensors.
