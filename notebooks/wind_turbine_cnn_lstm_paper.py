#!/usr/bin/env python
# coding: utf-8

# # Wind Turbine Fault Detection Based on CNN-LSTM
# 
# Based on Qi et al., *Research on Wind Turbine Fault Detection Based on CNN-LSTM*, Energies 2024.
# 
# - **Dataset**: CARE_To_Compare — Wind Farm C (58 events, 238 sensors, 10-min SCADA)
# - **Task**: Binary classification (normal vs anomaly)
# - **Architecture**: CNN-LSTM Option 1 (paper Table 2)
# 
# ### Pipeline
# 
# | Step | Description |
# |------|-------------|
# | 1 | Data loading & event-level split (70/15/15) |
# | 2 | Min-Max normalization (train normal rows only) |
# | 3 | XGBoost feature selection (top 30%) |
# | 4 | Sliding windows (36 steps = 6h) with undersampling (train only) |
# | 5 | CNN-LSTM training (Adam, CrossEntropy, batch=600, epochs=500) |
# | 6 | Evaluation & comparison with standalone CNN and LSTM |
# 
# ### Laboratorio (maquina compartilhada)
# 
# - Antes de treinar: `nvidia-smi` — use a GPU indicada pelo professor.
# - Padroes: `LAB_CUDA_DEVICE` / `CUDA_VISIBLE_DEVICES` (padrao **1**), `LAB_CPU_THREADS` (padrao **6**).
# - Dataset: `CARE_To_Compare/Wind Farm C` na raiz do repo ou `CARE_WIND_FARM_C`.
# 

# ## 1. Setup & Configuration

# In[5]:


import os

# --- Laboratorio (maquina compartilhida): ANTES de importar torch ---
_lab_threads = os.environ.get("LAB_CPU_THREADS", "6").strip()
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, _lab_threads)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("LAB_CUDA_DEVICE", "1")

import gc
import json
import time
import warnings
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Recursos (lab): CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}  LAB_CPU_THREADS={_lab_threads}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy:   {np.__version__}")
print(f"Pandas:  {pd.__version__}")

# === Configuration matching Paper Option 1 ===
CONFIG = {
    # Paths
    "base_dir": "CARE_To_Compare/Wind Farm C",

    # Sliding window
    "window_size": 36,       # 36 timesteps = 6 hours at 10-min resolution

    # CNN-LSTM Option 1 (paper Table 2)
    "conv1_filters": 32,
    "conv1_kernel": 3,
    "pool1_size": 2,
    "conv2_filters": 64,
    "conv2_kernel": 2,
    "pool2_size": 3,
    "lstm1_units": 100,
    "lstm2_units": 80,
    "dropout_rate": 0.5,
    "dense1_units": 50,
    "dense2_units": 10,
    "num_classes": 2,        # binary: normal vs anomaly

    # Training (paper Table 3)
    "batch_size": 600,
    "epochs": 500,
    "learning_rate": 1e-3,
    "early_stopping_patience": 20,

    # Feature selection
    "xgb_top_pct": 0.30,    # top 30% features

    # Split
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    # Memory: cap total sliding windows to prevent OOM
    "max_total_windows": 100_000,
}

print("\nConfiguration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")


# ## 2. Data Loading & Preprocessing
# 
# Load 58 event CSVs from Wind Farm C. Events split into train/val/test **before** any preprocessing. Scaler statistics computed from normal rows of training events only.

# In[6]:


from pathlib import Path

base_dir_cfg = CONFIG["base_dir"]
cwd = Path.cwd()

# Resolve dataset path robustly for runs from workspace root or notebooks/ directory.
candidates = []
_env_care = os.environ.get("CARE_WIND_FARM_C", "").strip()
if _env_care:
    candidates.append(Path(os.path.normpath(_env_care)))
candidates.extend([
    cwd / base_dir_cfg,
    cwd.parent / base_dir_cfg,
    Path(base_dir_cfg),
])
base_dir_path = next((p.resolve() for p in candidates if (p / "event_info.csv").exists()), None)
if base_dir_path is None:
    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not locate CARE Wind Farm C folder. Checked:\n" + checked
    )

base_dir = str(base_dir_path)
CONFIG["base_dir"] = base_dir

event_info = pd.read_csv(os.path.join(base_dir, "event_info.csv"), sep=";")
feature_desc = pd.read_csv(os.path.join(base_dir, "feature_description.csv"), sep=";")

print(f"Using base_dir: {base_dir}")
print(f"Events: {len(event_info)}")
print(f"  Anomaly: {(event_info['event_label'] == 'anomaly').sum()}")
print(f"  Normal:  {(event_info['event_label'] == 'normal').sum()}")
print(f"\nSensors described: {len(feature_desc)}")
print(f"\nEvent info columns: {list(event_info.columns)}")
print(event_info.head(10).to_string())


# In[7]:


datasets_dir = os.path.join(base_dir, "datasets")
csv_files = sorted(
    [f for f in os.listdir(datasets_dir) if f.endswith(".csv")],
    key=lambda x: int(x.replace(".csv", ""))
)
event_label_map = dict(zip(event_info["event_id"], event_info["event_label"]))
event_start_map = dict(zip(event_info["event_id"], event_info["event_start_id"]))
event_end_map   = dict(zip(event_info["event_id"], event_info["event_end_id"]))
META_COLS = ["time_stamp", "asset_id", "id", "train_test", "status_type_id"]

# --- Discover sensor columns ---
sample_df = pd.read_csv(os.path.join(datasets_dir, csv_files[0]), sep=";", nrows=5)
orig_sensor_cols = [c for c in sample_df.columns if c not in META_COLS
               and sample_df[c].dtype in [np.float64, np.int64, np.float32]]
orig_dtype_map = {c: np.float32 for c in orig_sensor_cols}
del sample_df

# --- Identify Angle Features ---
if "is_angle" in feature_desc.columns:
    angle_sensors_set = set(feature_desc.loc[feature_desc['is_angle'] == True, 'sensor_name'].values)
else:
    angle_sensors_set = set()

angle_cols = [c for c in orig_sensor_cols if c in angle_sensors_set]
non_angle_cols = [c for c in orig_sensor_cols if c not in angle_sensors_set]

# Expanded columns (replacing angles with sin/cos)
sensor_cols = non_angle_cols.copy()
for c in angle_cols:
    sensor_cols.extend([f"{c}_sin", f"{c}_cos"])

print(f"Original sensor columns: {len(orig_sensor_cols)}")
print(f"Angle columns: {len(angle_cols)} -> Transformed into Sin/Cos")
print(f"Expanded sensor columns: {len(sensor_cols)}")

def load_event(csv_file, fill=True):
    """Load one event CSV. Applies sin/cos to angles."""
    event_id = int(csv_file.replace(".csv", ""))
    usecols = orig_sensor_cols + ["id"]
    df = pd.read_csv(os.path.join(datasets_dir, csv_file), sep=";",
                     dtype=orig_dtype_map, usecols=usecols)

    # Apply sin/cos
    for c in angle_cols:
        rad = df[c] * np.pi / 180.0
        df[f"{c}_sin"] = np.sin(rad).astype(np.float32)
        df[f"{c}_cos"] = np.cos(rad).astype(np.float32)
        df.drop(columns=[c], inplace=True)

    event_lbl = event_label_map.get(event_id, "unknown")
    df["label"] = np.int8(0)
    if event_lbl == "anomaly":
        start_id = event_start_map.get(event_id)
        end_id   = event_end_map.get(event_id)
        if start_id is not None and end_id is not None:
            df.loc[(df["id"] >= start_id) & (df["id"] <= end_id), "label"] = np.int8(1)
    df.drop(columns=["id"], inplace=True)
    if fill:
        df.ffill(inplace=True)
        df.bfill(inplace=True)
    df["event_id"] = np.int16(event_id)
    return df

# =============================================================
# SPLIT EVENT IDs FIRST (prevent data leakage)
# =============================================================
all_event_ids = np.array([int(f.replace(".csv", "")) for f in csv_files])
np.random.shuffle(all_event_ids)
n_tr = int(len(all_event_ids) * CONFIG["train_ratio"])
n_va = int(len(all_event_ids) * CONFIG["val_ratio"])

train_eids = set(all_event_ids[:n_tr])
val_eids   = set(all_event_ids[n_tr:n_tr + n_va])
test_eids  = set(all_event_ids[n_tr + n_va:])

train_csv_files = [f for f in csv_files if int(f.replace(".csv", "")) in train_eids]
val_csv_files   = [f for f in csv_files if int(f.replace(".csv", "")) in val_eids]
test_csv_files  = [f for f in csv_files if int(f.replace(".csv", "")) in test_eids]

print(f"\nEvent-level split (leakage-free):")
print(f"  Train events: {len(train_eids)} | Val events: {len(val_eids)} | Test events: {len(test_eids)}")
for split_name, split_eids in [("Train", train_eids), ("Val", val_eids), ("Test", test_eids)]:
    n_anom = sum(1 for eid in split_eids if event_label_map.get(eid) == "anomaly")
    n_norm = sum(1 for eid in split_eids if event_label_map.get(eid) == "normal")
    print(f"  {split_name:5s}: {n_anom} anomaly events, {n_norm} normal events")

# =============================================================
# PASS 1: Compute MinMaxScaler stats from TRAIN events only
# Uses chunked CSV reading to avoid large single allocations.
# =============================================================
gc.collect()  # free any leftover memory from previous runs
print("\n--- Pass 1: Computing scaler statistics (TRAIN events only, chunked) ---")
col_min = np.full(len(sensor_cols), np.inf, dtype=np.float64)
col_max = np.full(len(sensor_cols), -np.inf, dtype=np.float64)
total_rows = 0
total_anomaly = 0
CHUNK_SIZE_PASS1 = 10_000

for i, csv_file in enumerate(csv_files):
    event_id = int(csv_file.replace(".csv", ""))
    filepath = os.path.join(datasets_dir, csv_file)
    event_lbl = event_label_map.get(event_id, "unknown")
    start_id = event_start_map.get(event_id) if event_lbl == "anomaly" else None
    end_id   = event_end_map.get(event_id) if event_lbl == "anomaly" else None

    for chunk in pd.read_csv(filepath, sep=";", dtype=orig_dtype_map,
                             usecols=orig_sensor_cols + ["id"],
                             chunksize=CHUNK_SIZE_PASS1):

        # Apply sin/cos
        for c in angle_cols:
            rad = chunk[c] * np.pi / 180.0
            chunk[f"{c}_sin"] = np.sin(rad).astype(np.float32)
            chunk[f"{c}_cos"] = np.cos(rad).astype(np.float32)
            chunk.drop(columns=[c], inplace=True)

        # Reorder chunk to match sensor_cols strictly
        chunk_vals = chunk[sensor_cols].values

        ids = chunk["id"].values
        if start_id is not None and end_id is not None:
            anom_mask = (ids >= start_id) & (ids <= end_id)
        else:
            anom_mask = np.zeros(len(chunk), dtype=bool)

        total_rows += len(chunk)
        total_anomaly += int(anom_mask.sum())

        # Only use TRAIN events for scaler fitting (no data leakage)
        if event_id in train_eids:
            normal_mask = ~anom_mask
            if normal_mask.any():
                normal_vals = chunk_vals[normal_mask, :]
                col_min = np.minimum(col_min, np.nanmin(normal_vals, axis=0))
                col_max = np.maximum(col_max, np.nanmax(normal_vals, axis=0))
                del normal_vals
        del chunk, chunk_vals
    gc.collect()
    if (i + 1) % 20 == 0:
        print(f"  Scanned {i+1}/{len(csv_files)} files")

# Build scaler manually
col_range = col_max - col_min
col_range[col_range == 0] = 1.0  # avoid division by zero for constant features

print(f"\nTotal rows across all files: {total_rows:,}")
print(f"Anomaly rows: {total_anomaly:,}")
print(f"Normal rows: {total_rows - total_anomaly:,}")
print(f"Imbalance ratio: {(total_rows - total_anomaly) / max(total_anomaly, 1):.1f}:1")


# ## 3. Feature Selection (XGBoost)
# 
# Train XGBoost on a subsample from training events, then select the top 30% of features by importance (paper Section 3.2.3).

# In[8]:


# =============================================================
# PASS 2: Gather subsample for XGBoost feature selection
#         (TRAIN events only â€” no data leakage)
# =============================================================
print("--- Pass 2: Sampling rows for XGBoost (TRAIN events only) ---")
XGB_SAMPLE_TARGET = 100_000
rows_per_file = max(1, XGB_SAMPLE_TARGET // len(train_csv_files))
xgb_X_parts, xgb_y_parts = [], []

for i, csv_file in enumerate(train_csv_files):
    df = load_event(csv_file)
    vals = df[sensor_cols].values.astype(np.float32)
    labels = df["label"].values

    # Normalize using streaming scaler
    vals = (vals - col_min.astype(np.float32)) / col_range.astype(np.float32)

    # Sample rows from this file
    n_sample = min(rows_per_file, len(df))
    idx = np.random.choice(len(df), n_sample, replace=False)
    xgb_X_parts.append(vals[idx])
    xgb_y_parts.append(labels[idx])
    del df, vals, labels
    gc.collect()

X_xgb = np.concatenate(xgb_X_parts, axis=0)
y_xgb = np.concatenate(xgb_y_parts, axis=0)
del xgb_X_parts, xgb_y_parts
gc.collect()

print(f"XGBoost sample: {len(X_xgb):,} rows, {X_xgb.shape[1]} features")
print(f"  Class 0: {(y_xgb==0).sum():,}  Class 1: {(y_xgb==1).sum():,}")

n_pos = y_xgb.sum()
n_neg = len(y_xgb) - n_pos
scale_pos = n_neg / max(n_pos, 1)

xgb_model = XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos, eval_metric="logloss",
    random_state=SEED, n_jobs=-1, verbosity=0,
)
xgb_model.fit(X_xgb, y_xgb)

importances = xgb_model.feature_importances_
feature_importance = pd.DataFrame({
    "feature": sensor_cols, "importance": importances
}).sort_values("importance", ascending=False)

n_top = max(1, int(len(sensor_cols) * CONFIG["xgb_top_pct"]))
selected_features = feature_importance.head(n_top)["feature"].tolist()

print(f"\nTotal features: {len(sensor_cols)}")
print(f"Selected top {CONFIG['xgb_top_pct']*100:.0f}%: {len(selected_features)} features")

fig, ax = plt.subplots(figsize=(10, 8))
top30 = feature_importance.head(30)
ax.barh(range(len(top30)), top30["importance"].values)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(top30["feature"].values, fontsize=8)
ax.set_xlabel("Importance")
ax.set_title("Top 30 Features (XGBoost)")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# Update sensor_cols to selected only; recompute scaler arrays for selected features
sel_idx = [sensor_cols.index(f) for f in selected_features]
sensor_cols = selected_features
col_min = col_min[sel_idx]
col_max = col_max[sel_idx]
col_range = col_range[sel_idx]

del X_xgb, y_xgb, xgb_model
gc.collect()
print(f"\nWorking with {len(sensor_cols)} features from now on")


# ## 4. Sliding Windows & Undersampling
# 
# Sliding windows of 36 timesteps (6h). A window is labeled as anomaly if **any** timestep within it is anomalous. Normal windows are undersampled **only in training**; val/test keep their natural distribution.

# In[ ]:


# =============================================================
# PASS 3: Build lazy window index WITH undersampling (TRAIN only)
# Val/Test keep natural class distribution to avoid evaluation bias.
# Memory-efficient: stores flat arrays per event, builds windows on-the-fly.
# =============================================================
print("--- Pass 3: Building window index (undersampling ONLY train) ---")
W = CONFIG["window_size"]
n_features = len(sensor_cols)
col_min_f32 = col_min.astype(np.float32)
col_range_f32 = col_range.astype(np.float32)

# Phase 1: Quick scan to count anomaly/normal windows PER SPLIT
file_win_info = []
train_anom_win, train_norm_win = 0, 0
for csv_file in csv_files:
    event_id = int(csv_file.replace(".csv", ""))
    event_lbl = event_label_map.get(event_id, "unknown")
    is_train = event_id in train_eids
    id_col = pd.read_csv(os.path.join(datasets_dir, csv_file), sep=";",
                         usecols=["id"])
    n = len(id_col)
    if n < W:
        file_win_info.append((csv_file, event_id, 0, 0, is_train))
        del id_col
        continue
    labels = np.zeros(n, dtype=np.int8)
    if event_lbl == "anomaly":
        start_id = event_start_map.get(event_id)
        end_id   = event_end_map.get(event_id)
        if start_id is not None and end_id is not None:
            mask = (id_col["id"].values >= start_id) & (id_col["id"].values <= end_id)
            labels[mask] = 1
    cum_labels = np.concatenate([[0], np.cumsum(labels)])
    wl = (cum_labels[W:] - cum_labels[:n - W + 1] > 0).astype(np.int8)
    n_a = int((wl == 1).sum())
    n_n = int((wl == 0).sum())
    if is_train:
        train_anom_win += n_a
        train_norm_win += n_n
    file_win_info.append((csv_file, event_id, n_a, n_n, is_train))
    del id_col, labels, wl

print(f"Train windows before undersample: anom={train_anom_win:,}  norm={train_norm_win:,}")

# Balance classes ONLY within training events
if train_anom_win > 0 and train_norm_win > train_anom_win:
    keep_frac = train_anom_win / train_norm_win
else:
    keep_frac = 1.0
print(f"Train normal keep fraction: {keep_frac:.6f}")

# Phase 2: Load flat arrays per event & build compact window index
# Instead of materializing (N, W, F) tensor, store flat (rows, F) per event
# and a compact index of (event_id, start_row, label) per window.
group_arrays = {}   # event_id -> (rows, n_features) float32
all_eids, all_starts, all_labels, all_splits = [], [], [], []
ptr = 0

for i, (csv_file, event_id, n_a, n_n, is_train) in enumerate(file_win_info):
    if n_a + n_n == 0:
        continue

    df = load_event(csv_file)
    features = df[sensor_cols].values.astype(np.float32)
    labels = df["label"].values
    del df

    # Normalize
    features = (features - col_min_f32) / col_range_f32
    group_arrays[event_id] = features

    n_win = len(features) - W + 1
    cum_labels = np.concatenate([[0], np.cumsum(labels)])
    win_labels = (cum_labels[W:] - cum_labels[:n_win] > 0).astype(np.int8)

    anom_idx = np.where(win_labels == 1)[0]
    norm_idx = np.where(win_labels == 0)[0]

    if is_train:
        n_norm_keep = int(len(norm_idx) * keep_frac)
        if n_norm_keep > 0 and len(norm_idx) > 0:
            norm_keep = np.random.choice(norm_idx, n_norm_keep, replace=False)
        else:
            norm_keep = np.array([], dtype=np.intp)
        split_tag = 0  # train
    else:
        norm_keep = norm_idx
        if event_id in val_eids:
            split_tag = 1  # val
        else:
            split_tag = 2  # test

    keep_idx = np.sort(np.concatenate([anom_idx, norm_keep]))
    n_keep = len(keep_idx)

    if n_keep > 0:
        all_eids.append(np.full(n_keep, event_id, dtype=np.int32))
        all_starts.append(keep_idx.astype(np.int32))
        all_labels.append(win_labels[keep_idx])
        all_splits.append(np.full(n_keep, split_tag, dtype=np.int8))
        ptr += n_keep

    del features, labels, win_labels, keep_idx
    gc.collect()

    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{len(csv_files)} files ({ptr:,} windows indexed)")

all_eids   = np.concatenate(all_eids)
all_starts = np.concatenate(all_starts)
all_labels = np.concatenate(all_labels)
all_splits = np.concatenate(all_splits)

ga_mem = sum(a.nbytes for a in group_arrays.values()) / 1e9
materialized_gb = ptr * W * n_features * 4 / 1e9

print(f"\nTotal windows indexed: {ptr:,}")
print(f"Labels: 0={int((all_labels==0).sum()):,}  1={int((all_labels==1).sum()):,}")
print(f"Memory (flat arrays): {ga_mem:.2f} GB  vs  {materialized_gb:.1f} GB if materialized")
print(f"\nNOTA: Undersampling aplicado SOMENTE em eventos de treino.")
print(f"      Val/Test mantem distribuicao natural de classes.")


# ## 5. Train / Val / Test Split

# In[ ]:


# --- Split window index by event-level split (already defined) ---
tr_mask = (all_splits == 0)
va_mask = (all_splits == 1)
te_mask = (all_splits == 2)

eid_tr, start_tr, y_train = all_eids[tr_mask], all_starts[tr_mask], all_labels[tr_mask]
eid_va, start_va, y_val   = all_eids[va_mask], all_starts[va_mask], all_labels[va_mask]
eid_te, start_te, y_test  = all_eids[te_mask], all_starts[te_mask], all_labels[te_mask]

del all_eids, all_starts, all_labels, all_splits, tr_mask, va_mask, te_mask
gc.collect()

for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    print(f"{name:5s}: {len(y):,} windows | 0={int((y==0).sum()):,}  1={int((y==1).sum()):,}")


# In[ ]:


# --- Class distribution per split ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, y) in zip(axes, [('Train', y_train), ('Val', y_val), ('Test', y_test)]):
    unique, counts = np.unique(y, return_counts=True)
    labels_map = {0: 'Normal', 1: 'Anomaly'}
    colors = ['#4C72B0', '#C44E52']
    bars = ax.bar([labels_map.get(int(u), str(u)) for u in unique],
                  counts, color=colors[:len(unique)], edgecolor='black', linewidth=0.5)
    total = counts.sum()
    for bar, val in zip(bars, counts):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
    ax.set_title(f'{name}', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of windows')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Class Distribution per Split (Undersampling ONLY on Train)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nNOTA: Train foi balanceado via undersampling de janelas normais.")
print("      Val e Test mantem a distribuicao NATURAL de classes.")


# ## 6. PyTorch DataLoaders

# In[ ]:


class WindowDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset: builds each window on-the-fly from flat arrays."""
    def __init__(self, group_arrays, eids, starts, labels, window_size):
        self.ga     = group_arrays
        self.eids   = eids
        self.starts = starts
        self.labels = labels
        self.ws     = window_size

    def __len__(self):
        return len(self.eids)

    def __getitem__(self, idx):
        e = int(self.eids[idx])
        s = int(self.starts[idx])
        x = torch.from_numpy(self.ga[e][s:s + self.ws].copy())
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y

BS = CONFIG["batch_size"]

train_dataset = WindowDataset(group_arrays, eid_tr, start_tr, y_train, W)
val_dataset   = WindowDataset(group_arrays, eid_va, start_va, y_val, W)
test_dataset  = WindowDataset(group_arrays, eid_te, start_te, y_test, W)

train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True,
                          drop_last=False, num_workers=0, pin_memory=torch.cuda.is_available())
val_loader   = DataLoader(val_dataset, batch_size=BS, shuffle=False,
                          drop_last=False, num_workers=0, pin_memory=torch.cuda.is_available())
test_loader  = DataLoader(test_dataset, batch_size=BS, shuffle=False,
                          drop_last=False, num_workers=0, pin_memory=torch.cuda.is_available())

print(f"DataLoaders ready (lazy, on-the-fly) | batch_size={BS}")
print(f"Input shape per sample: ({W}, {n_features})")
print(f"  Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")


# ## 7. CNN-LSTM Architecture (Option 1)
# 
# | Layer | Configuration |
# |-------|---------------|
# | Conv1D + SELU | 32 filters, kernel=3 → MaxPool(2) |
# | Conv1D + SELU | 64 filters, kernel=2 → MaxPool(3) |
# | LSTM | 100 units → 80 units |
# | Dropout | 50% |
# | Dense + SELU | 50 → 10 → 2 (Softmax) |

# In[ ]:


class CNNLSTM(nn.Module):
    """CNN-LSTM fault detection model following Paper Option 1 (Table 2)."""

    def __init__(self, n_features, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        self.lstm1 = nn.LSTM(64, 100, batch_first=True)
        self.lstm2 = nn.LSTM(100, 80, batch_first=True)
        # Use Standard Dropout instead of AlphaDropout because the output of LSTM is not SELU-normalized
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(80, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, n_classes)
        self.selu = nn.SELU()

        # Lecun normal init â€” required for SELU + AlphaDropout
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)          # (batch, features, timesteps)
        x = self.selu(self.conv1(x))
        x = self.pool1(x)
        x = self.selu(self.conv2(x))
        x = self.pool2(x)
        x = x.permute(0, 2, 1)          # (batch, T', 64) for LSTM
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]                 # last timestep
        x = self.dropout(x)
        x = self.selu(self.fc1(x))
        x = self.selu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNNLSTM(n_features=n_features, n_classes=CONFIG["num_classes"]).to(DEVICE)
total_p = sum(p.numel() for p in model.parameters())
print(f"CNN-LSTM Model: {total_p:,} parameters")
print(model)


# In[ ]:


def train_model(model, train_loader, val_loader, config, device, model_name="CNN-LSTM"):
    """Training loop with early stopping. Returns history dict."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(config["epochs"]):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * len(y_b)
            correct += (out.argmax(1) == y_b).sum().item()
            total += len(y_b)
        tr_loss, tr_acc = run_loss / total, correct / total

        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                out = model(X_b)
                vl += criterion(out, y_b).item() * len(y_b)
                vc += (out.argmax(1) == y_b).sum().item()
                vt += len(y_b)
        va_loss, va_acc = vl / vt, vc / vt

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[{model_name}] Epoch {epoch+1:3d}/{config['epochs']} | "
                  f"TrLoss:{tr_loss:.4f} TrAcc:{tr_acc:.4f} | "
                  f"VaLoss:{va_loss:.4f} VaAcc:{va_acc:.4f} | "
                  f"Pat:{patience_counter}/{config['early_stopping_patience']}")

        if patience_counter >= config["early_stopping_patience"]:
            print(f"[{model_name}] Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return history


# ## 8. Training

# In[ ]:


print("=" * 60)
print("Training CNN-LSTM Model")
print("=" * 60)
cnn_lstm_history = train_model(model, train_loader, val_loader, CONFIG, DEVICE, "CNN-LSTM")


# ## 9. Training Curves

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(cnn_lstm_history["train_acc"], label="Train")
axes[0].plot(cnn_lstm_history["val_acc"], label="Validation")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].set_title("CNN-LSTM Accuracy"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(cnn_lstm_history["train_loss"], label="Train")
axes[1].plot(cnn_lstm_history["val_loss"], label="Validation")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].set_title("CNN-LSTM Loss"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()


# In[ ]:


def evaluate_model(model, loader, device, threshold=0.5):
    """Returns (y_true, y_pred, y_proba). Uses custom threshold for anomaly class predictions."""
    model.eval()
    preds, labels, proba = [], [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            out = model(X_b.to(device))
            p = torch.softmax(out, dim=1)

            p1 = p[:, 1].cpu().numpy()
            pred = (p1 >= threshold).astype(int)

            preds.extend(pred)
            labels.extend(y_b.numpy())
            proba.extend(p1)
    return np.array(labels), np.array(preds), np.array(proba)

def find_best_threshold(y_true, y_proba):
    """Finds the decision threshold that maximizes F1-Score, important for artificially balanced training data."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    # add small epsilon to avoid division by zero
    f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    print(f"  [Validation] Optimal Threshold: {best_thresh:.4f} \u2192 F1-Score: {best_f1:.4f}")
    return best_thresh

def print_metrics(y_true, y_pred, y_proba, name="Model"):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try: auc_roc = roc_auc_score(y_true, y_proba)
    except: auc_roc = 0.0
    print(f"\n{'='*50}\n  {name} - Test Results\n{'='*50}")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print(f"  AUC-ROC:   {auc_roc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Normal","Anomaly"], zero_division=0))
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc_roc": auc_roc}


# In[ ]:


# 1. Optimize threshold on Validation subset
y_true_val, _, y_proba_val = evaluate_model(model, val_loader, DEVICE, threshold=0.5) # threshold doesn't matter for proba
best_thresh_cnnlstm = find_best_threshold(y_true_val, y_proba_val)

# 2. Evaluate on Test subset with optimized threshold
y_true_cnnlstm, y_pred_cnnlstm, y_proba_cnnlstm = evaluate_model(model, test_loader, DEVICE, threshold=best_thresh_cnnlstm)
metrics_cnnlstm = print_metrics(y_true_cnnlstm, y_pred_cnnlstm, y_proba_cnnlstm, "CNN-LSTM")

fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_true_cnnlstm, y_pred_cnnlstm)
ConfusionMatrixDisplay(cm, display_labels=["Normal","Anomaly"]).plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("CNN-LSTM Confusion Matrix"); plt.tight_layout(); plt.show()


# ## 10. Comparison Models
# 
# Standalone CNN and LSTM trained with the same data and hyperparameters (paper Section 4.2).

# In[ ]:


class CNNOnly(nn.Module):
    def __init__(self, n_features, n_classes=2, seq_len=36):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        self.selu = nn.SELU()
        self.dropout = nn.AlphaDropout(0.5)  # AlphaDropout preserva propriedade auto-normalizante do SELU
        with torch.no_grad():
            d = torch.zeros(1, n_features, seq_len)
            d = self.pool1(self.selu(self.conv1(d)))
            d = self.pool2(self.selu(self.conv2(d)))
            flat = d.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flat, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, n_classes)

        # Lecun normal init â€” required for SELU + AlphaDropout
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.selu(self.conv1(x)); x = self.pool1(x)
        x = self.selu(self.conv2(x)); x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.selu(self.fc1(x)); x = self.selu(self.fc2(x))
        return self.fc3(x)

class LSTMOnly(nn.Module):
    def __init__(self, n_features, n_classes=2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, 100, batch_first=True)
        self.lstm2 = nn.LSTM(100, 80, batch_first=True)
        # Use Standard Dropout instead of AlphaDropout because the output of LSTM is not SELU-normalized
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(80, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, n_classes)
        self.selu = nn.SELU()

        # Lecun normal init â€” required for SELU + AlphaDropout
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x, _ = self.lstm1(x); x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.selu(self.fc1(x)); x = self.selu(self.fc2(x))
        return self.fc3(x)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("=" * 60, "\nTraining Standalone CNN\n" + "=" * 60)
cnn_model = CNNOnly(n_features, CONFIG["num_classes"], W).to(DEVICE)
cnn_history = train_model(cnn_model, train_loader, val_loader, CONFIG, DEVICE, "CNN")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "=" * 60, "\nTraining Standalone LSTM\n" + "=" * 60)
lstm_model = LSTMOnly(n_features, CONFIG["num_classes"]).to(DEVICE)
lstm_history = train_model(lstm_model, train_loader, val_loader, CONFIG, DEVICE, "LSTM")


# ## 11. Evaluation

# In[ ]:


# 1. CNN evaluation
y_true_val_cnn, _, y_proba_val_cnn = evaluate_model(cnn_model, val_loader, DEVICE)
best_thresh_cnn = find_best_threshold(y_true_val_cnn, y_proba_val_cnn)
y_true_cnn, y_pred_cnn, y_proba_cnn = evaluate_model(cnn_model, test_loader, DEVICE, threshold=best_thresh_cnn)
metrics_cnn = print_metrics(y_true_cnn, y_pred_cnn, y_proba_cnn, "CNN")

# 2. LSTM evaluation
y_true_val_lstm, _, y_proba_val_lstm = evaluate_model(lstm_model, val_loader, DEVICE)
best_thresh_lstm = find_best_threshold(y_true_val_lstm, y_proba_val_lstm)
y_true_lstm, y_pred_lstm, y_proba_lstm = evaluate_model(lstm_model, test_loader, DEVICE, threshold=best_thresh_lstm)
metrics_lstm = print_metrics(y_true_lstm, y_pred_lstm, y_proba_lstm, "LSTM")


# In[ ]:


# Summary table
comparison_df = pd.DataFrame({
    "Model": ["CNN", "LSTM", "CNN-LSTM"],
    "Accuracy (%)": [metrics_cnn["accuracy"]*100, metrics_lstm["accuracy"]*100, metrics_cnnlstm["accuracy"]*100],
    "Precision (%)": [metrics_cnn["precision"]*100, metrics_lstm["precision"]*100, metrics_cnnlstm["precision"]*100],
    "Recall (%)": [metrics_cnn["recall"]*100, metrics_lstm["recall"]*100, metrics_cnnlstm["recall"]*100],
    "F1-Score (%)": [metrics_cnn["f1"]*100, metrics_lstm["f1"]*100, metrics_cnnlstm["f1"]*100],
    "AUC-ROC": [metrics_cnn["auc_roc"], metrics_lstm["auc_roc"], metrics_cnnlstm["auc_roc"]],
})
print("Model Comparison (Paper Table 4 style):")
print(comparison_df.round(2).to_string())


# In[ ]:


# Bar charts (like paper Figures 7-10)
models_names = ["CNN", "LSTM", "CNN-LSTM"]
metrics_list = [metrics_cnn, metrics_lstm, metrics_cnnlstm]
metric_keys = ["precision", "recall", "f1", "accuracy"]
metric_titles = ["Precision", "Recall", "F1-Score", "Accuracy"]
colors = ["#4C72B0", "#55A868", "#C44E52"]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, (key, title) in enumerate(zip(metric_keys, metric_titles)):
    ax = axes.ravel()[idx]
    values = [m[key] * 100 for m in metrics_list]
    bars = ax.bar(models_names, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel(f"{title} (%)"); ax.set_title(f"Comparison of {title}")
    ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.suptitle("Model Performance Comparison (Paper Section 4.3)", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.show()


# ## 12. Confusion Matrices & ROC Curves

# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, yt, yp, name in [
    (axes[0], y_true_cnn, y_pred_cnn, "CNN"),
    (axes[1], y_true_lstm, y_pred_lstm, "LSTM"),
    (axes[2], y_true_cnnlstm, y_pred_cnnlstm, "CNN-LSTM"),
]:
    cm = confusion_matrix(yt, yp)
    ConfusionMatrixDisplay(cm, display_labels=["Normal","Anomaly"]).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(name)
plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(8, 6))
for yt, yp, name, color in [
    (y_true_cnn, y_proba_cnn, "CNN", "#4C72B0"),
    (y_true_lstm, y_proba_lstm, "LSTM", "#55A868"),
    (y_true_cnnlstm, y_proba_cnnlstm, "CNN-LSTM", "#C44E52"),
]:
    fpr, tpr, _ = roc_curve(yt, yp)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves"); ax.legend(loc="lower right"); ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.show()


# In[ ]:


# --- Precision-Recall Curves ---
fig, ax = plt.subplots(figsize=(8, 6))
for yt, yp, name, color in [
    (y_true_cnn, y_proba_cnn, "CNN", "#4C72B0"),
    (y_true_lstm, y_proba_lstm, "LSTM", "#55A868"),
    (y_true_cnnlstm, y_proba_cnnlstm, "CNN-LSTM", "#C44E52"),
]:
    prec, rec, _ = precision_recall_curve(yt, yp)
    ap = average_precision_score(yt, yp)
    ax.plot(rec, prec, color=color, lw=2, label=f"{name} (AP={ap:.3f})")

baseline = y_true_cnnlstm.sum() / len(y_true_cnnlstm)
ax.axhline(y=baseline, color='k', linestyle='--', lw=1, alpha=0.5,
           label=f"Baseline (prevalence={baseline:.4f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves"); ax.legend(loc="upper right")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05]); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()


# ## 13. Training Curves Comparison

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for hist, name, color in [
    (cnn_history, "CNN", "#4C72B0"),
    (lstm_history, "LSTM", "#55A868"),
    (cnn_lstm_history, "CNN-LSTM", "#C44E52"),
]:
    axes[0].plot(hist["val_acc"], label=name, color=color)
    axes[1].plot(hist["val_loss"], label=name, color=color)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val Accuracy")
axes[0].set_title("Validation Accuracy"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val Loss")
axes[1].set_title("Validation Loss"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()


# In[ ]:


save_dir = "../resultados/results_cnn_lstm_paper"
os.makedirs(save_dir, exist_ok=True)

torch.save(model.state_dict(), os.path.join(save_dir, "cnn_lstm_model.pth"))
torch.save(cnn_model.state_dict(), os.path.join(save_dir, "cnn_model.pth"))
torch.save(lstm_model.state_dict(), os.path.join(save_dir, "lstm_model.pth"))

# Save scaler arrays for reproducibility
np.savez(os.path.join(save_dir, "scaler.npz"),
         col_min=col_min, col_max=col_max, col_range=col_range)

results = {
    "selected_features": sensor_cols,
    "n_features": len(sensor_cols),
    "window_size": W,
    "undersampling": {
        "scope": "train_only",
        "keep_frac": float(keep_frac),
        "note": "Val/Test mantêm distribuição natural de classes"
    },
    "split": {
        "train_eids": sorted([int(e) for e in train_eids]),
        "val_eids": sorted([int(e) for e in val_eids]),
        "test_eids": sorted([int(e) for e in test_eids]),
    },
    "metrics": {
        "CNN": metrics_cnn,
        "LSTM": metrics_lstm,
        "CNN-LSTM": metrics_cnnlstm,
    }
}
with open(os.path.join(save_dir, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to '{save_dir}/':")
for fn in sorted(os.listdir(save_dir)):
    sz = os.path.getsize(os.path.join(save_dir, fn)) / 1024
    print(f"  {fn}: {sz:.1f} KB")


# ---
# 
# ## Conclusion
# 
# Implementation of the CNN-LSTM pipeline from Qi et al. (Energies 2024) adapted for CARE_To_Compare Wind Farm C.
# 
# **Key design decisions:**
# - Binary classification instead of 5-class
# - Event-level split prevents data leakage
# - Scaler fitted on normal training rows only
# - Undersampling applied only to training windows
# - Window labeled as anomaly if any timestep is anomalous
# - Streaming pipeline to handle 3M+ rows without OOM
