# === Cell 1: ConfiguraÃ§Ã£o e Imports ===
import os, sys, gc, json, warnings, random
import numpy as np
import pandas as pd

# Lab env
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try: tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError: pass

from tensorflow import keras
from tensorflow.keras import layers
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# Import shared utilities
sys.path.insert(0, os.path.dirname(os.path.abspath("__file__")))
from induced_failure_utils import *

PROJECT_ROOT = resolve_project_root()
DATASETS_DIR, EVENT_INFO_PATH, FEATURE_DESC_PATH = resolve_care_paths(PROJECT_ROOT)
ARTIFACTS_DIR = str(PROJECT_ROOT / "resultados" / "04_classifier_induced_failure")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

N_OPTUNA_TRIALS = int(os.environ.get("N_OPTUNA_TRIALS", "30"))
INJECTION_FRACTION = 0.07  # 7% of normal data â†’ induced anomaly

print(f"TensorFlow: {tf.__version__}")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATASETS_DIR: {DATASETS_DIR}")
print(f"N_OPTUNA_TRIALS: {N_OPTUNA_TRIALS}")

# === Cell 2: IngestÃ£o de Dados e Split por Evento ===

# Load all CSVs
df_raw = load_care_csvs(DATASETS_DIR)
event_info = load_event_info(EVENT_INFO_PATH)
feature_desc = load_feature_desc(FEATURE_DESC_PATH)
df_raw = merge_event_labels(df_raw, event_info)

# Identify features
FEATURE_COLS = infer_feature_columns(df_raw)
print(f"Features numÃ©ricas: {len(FEATURE_COLS)}")
print(f"Total amostras: {len(df_raw):,}")

# Event-level split (leakage-free)
event_label_map = dict(zip(event_info["event_id"], event_info["event_label"]))
all_event_ids = sorted(df_raw["source_file"].unique())
np.random.seed(SEED)
np.random.shuffle(all_event_ids)

n_tr = int(len(all_event_ids) * 0.70)
n_va = int(len(all_event_ids) * 0.15)

train_files = set(all_event_ids[:n_tr])
val_files = set(all_event_ids[n_tr:n_tr + n_va])
test_files = set(all_event_ids[n_tr + n_va:])

df_train = df_raw[df_raw["source_file"].isin(train_files)].copy()
df_val = df_raw[df_raw["source_file"].isin(val_files)].copy()
df_test = df_raw[df_raw["source_file"].isin(test_files)].copy()

del df_raw; gc.collect()

for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    y = infer_binary_labels(df)
    n_anom = y.sum()
    n_norm = len(y) - n_anom
    print(f"{name:5s}: {len(df):>10,} samples | Normal: {n_norm:>10,} | Anomaly: {n_anom:>8,}")

# === Cell 3: PrÃ©-processamento + SeleÃ§Ã£o de Features ===

# Extract feature matrices
X_train_raw = df_train[FEATURE_COLS].values.astype(np.float32)
X_val_raw = df_val[FEATURE_COLS].values.astype(np.float32)
X_test_raw = df_test[FEATURE_COLS].values.astype(np.float32)

y_train_binary = infer_binary_labels(df_train)
y_val_binary = infer_binary_labels(df_val)
y_test_binary = infer_binary_labels(df_test)

# Fit preprocessor on TRAIN NORMAL only
train_normal_mask = y_train_binary == 0
preprocessor = build_preprocessor()
preprocessor.fit(X_train_raw[train_normal_mask])

X_train_proc = preprocessor.transform(X_train_raw).astype(np.float32)
X_val_proc = preprocessor.transform(X_val_raw).astype(np.float32)
X_test_proc = preprocessor.transform(X_test_raw).astype(np.float32)

del X_train_raw, X_val_raw, X_test_raw; gc.collect()

# Unsupervised feature selection (fit on train only)
sel_indices, sel_names = select_features_unsupervised(
    X_train_proc[train_normal_mask], FEATURE_COLS
)
print(f"Features selecionadas: {len(sel_indices)} de {len(FEATURE_COLS)}")

X_train_sel = X_train_proc[:, sel_indices]
X_val_sel = X_val_proc[:, sel_indices]
X_test_sel = X_test_proc[:, sel_indices]

del X_train_proc, X_val_proc, X_test_proc; gc.collect()
print(f"Shapes: Train={X_train_sel.shape}, Val={X_val_sel.shape}, Test={X_test_sel.shape}")

# === Cell 4: InjeÃ§Ã£o de Falhas SintÃ©ticas (somente TREINO) ===

# Separate normal and anomaly training data
X_train_normal = X_train_sel[train_normal_mask]
X_train_anomaly = X_train_sel[~train_normal_mask]

print(f"Train normal: {len(X_train_normal):,}")
print(f"Train anomaly (real): {len(X_train_anomaly):,}")

# Inject synthetic failures into normal data
X_aug, y_aug, types_aug = inject_synthetic_failures(
    X_train_normal,
    feature_names=sel_names,
    injection_fraction=INJECTION_FRACTION,
    seed=SEED,
)

# Build 3-class training set: 0=Normal, 1=Real Anomaly, 2=Induced Anomaly
X_train_3class = np.concatenate([X_aug, X_train_anomaly], axis=0)
y_train_3class = np.concatenate([
    y_aug,  # 0=normal, 2=induced
    np.ones(len(X_train_anomaly), dtype=np.int32),  # 1=real anomaly
])

# Shuffle
perm = np.random.RandomState(SEED).permutation(len(X_train_3class))
X_train_3class = X_train_3class[perm]
y_train_3class = y_train_3class[perm]

for cls, name in [(0, "Normal"), (1, "Real Anomaly"), (2, "Induced Anomaly")]:
    n = (y_train_3class == cls).sum()
    print(f"  Class {cls} ({name}): {n:,} ({100*n/len(y_train_3class):.1f}%)")

# Type distribution of induced anomalies
from collections import Counter
type_counts = Counter(types_aug[types_aug != "none"])
print(f"\nInduced anomaly types: {dict(type_counts)}")

del X_aug, y_aug, types_aug, X_train_normal, X_train_anomaly; gc.collect()

# === Cell 5: Optuna â€” OtimizaÃ§Ã£o de HiperparÃ¢metros ===

INPUT_DIM = X_train_3class.shape[1]

def build_classifier(input_dim, n_layers, hidden_units, dropout_rate, learning_rate):
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for i in range(n_layers):
        units = max(16, hidden_units // (2 ** i))
        x = layers.Dense(units, kernel_initializer="he_normal")(x)
        x = layers.PReLU()(x)
        x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# Class weights for imbalance
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train_3class)
cw = compute_class_weight("balanced", classes=classes, y=y_train_3class)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}
print(f"Class weights: {class_weight_dict}")

def objective_classifier(trial):
    keras.backend.clear_session(); gc.collect()
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_units = trial.suggest_int("hidden_units", 64, 256, step=32)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    model = build_classifier(INPUT_DIM, n_layers, hidden_units, dropout_rate, lr)

    # Internal val split from training data (temporal)
    n_split = int(len(X_train_3class) * 0.85)
    Xtr, Xvl = X_train_3class[:n_split], X_train_3class[n_split:]
    ytr, yvl = y_train_3class[:n_split], y_train_3class[n_split:]

    model.fit(Xtr, ytr, validation_data=(Xvl, yvl), epochs=30,
              batch_size=128, shuffle=True, verbose=0, class_weight=class_weight_dict,
              callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)])

    # Evaluate on actual validation set (binary: real anomaly = 1, rest = 0)
    proba = model.predict(X_val_sel, verbose=0)
    # P(real anomaly) = class 1 probability
    y_pred_binary = (proba[:, 1] > 0.5).astype(int)
    f1 = f1_score(y_val_binary, y_pred_binary, zero_division=0)
    del model; keras.backend.clear_session(); gc.collect()
    return float(f1)

sampler = TPESampler(seed=SEED)
study = optuna.create_study(direction="maximize", sampler=sampler, study_name="classifier_induced")
study.optimize(objective_classifier, n_trials=N_OPTUNA_TRIALS)

best_params = study.best_trial.params
print(f"\nBest params: {best_params}")
print(f"Best F1 (val): {study.best_value:.4f}")

# === Cell 6: Treinamento Final + Baseline ===

keras.backend.clear_session(); gc.collect()

# --- Final model WITH induced anomalies ---
final_model = build_classifier(
    INPUT_DIM,
    n_layers=best_params["n_layers"],
    hidden_units=best_params["hidden_units"],
    dropout_rate=best_params["dropout_rate"],
    learning_rate=best_params["learning_rate"],
)

CKPT_PATH = os.path.join(ARTIFACTS_DIR, "classifier_best.keras")
history = final_model.fit(
    X_train_3class, y_train_3class,
    validation_data=(X_val_sel, np.where(y_val_binary == 1, 1, 0)),
    epochs=100, batch_size=128, shuffle=True, verbose=1,
    class_weight=class_weight_dict,
    callbacks=[
        keras.callbacks.EarlyStopping("val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(CKPT_PATH, "val_loss", save_best_only=True, verbose=1),
    ],
)

# --- Baseline model WITHOUT induced anomalies (2-class: normal vs real anomaly) ---
X_baseline = np.concatenate([X_train_sel[train_normal_mask], X_train_sel[~train_normal_mask]])
y_baseline = np.concatenate([
    np.zeros(train_normal_mask.sum(), dtype=np.int32),
    np.ones((~train_normal_mask).sum(), dtype=np.int32),
])
perm_b = np.random.RandomState(SEED).permutation(len(X_baseline))
X_baseline, y_baseline = X_baseline[perm_b], y_baseline[perm_b]

cw_baseline = compute_class_weight("balanced", classes=np.array([0,1]), y=y_baseline)
cw_baseline_dict = {0: float(cw_baseline[0]), 1: float(cw_baseline[1])}

baseline_model = build_classifier(
    INPUT_DIM,
    n_layers=best_params["n_layers"],
    hidden_units=best_params["hidden_units"],
    dropout_rate=best_params["dropout_rate"],
    learning_rate=best_params["learning_rate"],
)
# Rebuild last layer for 2-class
inputs_b = keras.Input(shape=(INPUT_DIM,))
x_b = inputs_b
for i in range(best_params["n_layers"]):
    units = max(16, best_params["hidden_units"] // (2 ** i))
    x_b = layers.Dense(units, kernel_initializer="he_normal")(x_b)
    x_b = layers.PReLU()(x_b)
    x_b = layers.Dropout(best_params["dropout_rate"])(x_b)
out_b = layers.Dense(2, activation="softmax")(x_b)
baseline_model = keras.Model(inputs_b, out_b)
baseline_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
    loss="sparse_categorical_crossentropy", metrics=["accuracy"],
)

baseline_history = baseline_model.fit(
    X_baseline, y_baseline,
    validation_data=(X_val_sel, y_val_binary),
    epochs=100, batch_size=128, shuffle=True, verbose=0,
    class_weight=cw_baseline_dict,
    callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=10, restore_best_weights=True)],
)

print(f"Final model epochs: {len(history.history['loss'])}")
print(f"Baseline model epochs: {len(baseline_history.history['loss'])}")

# === Cell 7: Threshold Ã“timo na ValidaÃ§Ã£o ===

# --- Induced model: use P(real anomaly) as score ---
proba_val = final_model.predict(X_val_sel, verbose=0)
scores_val = proba_val[:, 1]  # P(class=1, real anomaly)

# Find best threshold on validation
precisions, recalls, thresholds = precision_recall_curve(y_val_binary, scores_val)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-12)
best_idx = np.argmax(f1_scores)
best_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])
print(f"Best threshold (val F1): {best_threshold:.4f}")
print(f"Val F1 at best threshold: {f1_scores[best_idx]:.4f}")

# --- Baseline: same process ---
proba_val_base = baseline_model.predict(X_val_sel, verbose=0)
scores_val_base = proba_val_base[:, 1]
prec_b, rec_b, thr_b = precision_recall_curve(y_val_binary, scores_val_base)
f1_b = 2 * prec_b * rec_b / (prec_b + rec_b + 1e-12)
best_idx_b = np.argmax(f1_b)
best_threshold_base = float(thr_b[min(best_idx_b, len(thr_b) - 1)])
print(f"\nBaseline threshold (val F1): {best_threshold_base:.4f}")
print(f"Baseline Val F1: {f1_b[best_idx_b]:.4f}")

# === Cell 8: AvaliaÃ§Ã£o no Teste + CARE Score ===

# --- Induced model on TEST ---
proba_test = final_model.predict(X_test_sel, verbose=0)
scores_test = proba_test[:, 1]
y_pred_test = (scores_test > best_threshold).astype(int)

metrics_induced = print_classification_report(
    y_test_binary, y_pred_test, scores_test,
    title="INDUCED MODEL â€” Test Set"
)

# --- Baseline on TEST ---
proba_test_base = baseline_model.predict(X_test_sel, verbose=0)
scores_test_base = proba_test_base[:, 1]
y_pred_test_base = (scores_test_base > best_threshold_base).astype(int)

metrics_baseline = print_classification_report(
    y_test_binary, y_pred_test_base, scores_test_base,
    title="BASELINE (no induced) â€” Test Set"
)

# --- CARE Score (induced model) ---
care_ds, care_summary = evaluate_care_per_dataset(
    df_test, y_test_binary, y_pred_test, scores_test
)
print("\n" + "="*60)
print("  CARE Score â€” Induced Model")
print("="*60)
print(care_summary.to_string(index=False))

# --- CARE Score (baseline) ---
care_ds_base, care_summary_base = evaluate_care_per_dataset(
    df_test, y_test_binary, y_pred_test_base, scores_test_base
)
print("\n" + "="*60)
print("  CARE Score â€” Baseline")
print("="*60)
print(care_summary_base.to_string(index=False))

# --- Comparison table ---
comp = pd.DataFrame([
    {"Model": "Baseline (no induced)", **metrics_baseline,
     "CARE": float(care_summary_base["CARE"].iloc[0])},
    {"Model": "With Induced Anomalies", **metrics_induced,
     "CARE": float(care_summary["CARE"].iloc[0])},
])
print("\n" + "="*60)
print("  COMPARISON")
print("="*60)
print(comp.to_string(index=False))

# === Cell 9: VisualizaÃ§Ãµes ===

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Training loss
axes[0,0].plot(history.history["loss"], label="Train")
axes[0,0].plot(history.history["val_loss"], label="Val")
axes[0,0].set_title("Loss (Induced Model)"); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

# 2. Confusion Matrix â€” Induced
cm = confusion_matrix(y_test_binary, y_pred_test)
axes[0,1].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        axes[0,1].text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=12)
axes[0,1].set_title("Confusion Matrix (Induced)")
axes[0,1].set_xlabel("Predicted"); axes[0,1].set_ylabel("True")
axes[0,1].set_xticks([0,1]); axes[0,1].set_yticks([0,1])
axes[0,1].set_xticklabels(["Normal","Anomaly"]); axes[0,1].set_yticklabels(["Normal","Anomaly"])

# 3. Confusion Matrix â€” Baseline
cm_b = confusion_matrix(y_test_binary, y_pred_test_base)
axes[0,2].imshow(cm_b, cmap="Oranges")
for i in range(2):
    for j in range(2):
        axes[0,2].text(j, i, f"{cm_b[i,j]:,}", ha="center", va="center", fontsize=12)
axes[0,2].set_title("Confusion Matrix (Baseline)")
axes[0,2].set_xlabel("Predicted"); axes[0,2].set_ylabel("True")
axes[0,2].set_xticks([0,1]); axes[0,2].set_yticks([0,1])
axes[0,2].set_xticklabels(["Normal","Anomaly"]); axes[0,2].set_yticklabels(["Normal","Anomaly"])

# 4. ROC Curve comparison
if len(np.unique(y_test_binary)) > 1:
    fpr, tpr, _ = roc_curve(y_test_binary, scores_test)
    fpr_b, tpr_b, _ = roc_curve(y_test_binary, scores_test_base)
    axes[1,0].plot(fpr, tpr, label=f"Induced (AUC={roc_auc_score(y_test_binary, scores_test):.3f})")
    axes[1,0].plot(fpr_b, tpr_b, label=f"Baseline (AUC={roc_auc_score(y_test_binary, scores_test_base):.3f})")
    axes[1,0].plot([0,1],[0,1],"k--",alpha=0.3)
    axes[1,0].set_title("ROC Curve"); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

# 5. PR Curve comparison
prec_i, rec_i, _ = precision_recall_curve(y_test_binary, scores_test)
prec_b2, rec_b2, _ = precision_recall_curve(y_test_binary, scores_test_base)
axes[1,1].plot(rec_i, prec_i, label="Induced")
axes[1,1].plot(rec_b2, prec_b2, label="Baseline")
axes[1,1].set_title("PR Curve"); axes[1,1].set_xlabel("Recall"); axes[1,1].set_ylabel("Precision")
axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

# 6. CARE sub-scores comparison
subscores = ["F1_2", "Acc", "EF1_2", "WS", "CARE"]
x = np.arange(len(subscores)); w = 0.35
vals_induced = [float(care_summary[s].iloc[0]) for s in subscores]
vals_baseline = [float(care_summary_base[s].iloc[0]) for s in subscores]
axes[1,2].bar(x - w/2, vals_induced, w, label="Induced", color="#4C72B0")
axes[1,2].bar(x + w/2, vals_baseline, w, label="Baseline", color="#C44E52")
axes[1,2].set_xticks(x); axes[1,2].set_xticklabels(subscores)
axes[1,2].set_ylim(0, 1); axes[1,2].set_title("CARE Sub-scores"); axes[1,2].legend()

plt.suptitle("Classifier â€” Induced vs Baseline Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS_DIR, "comparison_plots.png"), dpi=150, bbox_inches="tight")
plt.show()
print(f"Plots saved to {ARTIFACTS_DIR}")

# === Cell 10: ExportaÃ§Ã£o de Artefatos ===

# Save models
final_model.save(os.path.join(ARTIFACTS_DIR, "classifier_induced.keras"))

# Save metrics
all_metrics = {
    "notebook": "wind_turbine_classifier_induced_failure",
    "paradigma": "supervisionado",
    "modelo": "MLP Classifier 3-class (Normal/Real/Induced)",
    "dataset": "CARE_To_Compare Wind Farm C",
    "hiperparametros": best_params,
    "threshold": best_threshold,
    "injection_fraction": INJECTION_FRACTION,
    "metricas_teste": {
        "induced_model": metrics_induced,
        "baseline": metrics_baseline,
    },
    "metricas_CARE": {
        "induced": care_summary.iloc[0].to_dict(),
        "baseline": care_summary_base.iloc[0].to_dict(),
    },
}
with open(os.path.join(ARTIFACTS_DIR, "metricas.json"), "w") as f:
    json.dump(all_metrics, f, indent=2, default=str)

# Save CARE per-dataset
care_ds.to_csv(os.path.join(ARTIFACTS_DIR, "care_results.csv"), index=False)

# Save best params
with open(os.path.join(ARTIFACTS_DIR, "best_params.json"), "w") as f:
    json.dump(best_params, f, indent=2)

# Save comparison
comp.to_csv(os.path.join(ARTIFACTS_DIR, "comparison.csv"), index=False)

print(f"Artefatos exportados em: {ARTIFACTS_DIR}")
print("\nArquivos gerados:")
for f in sorted(os.listdir(ARTIFACTS_DIR)):
    size = os.path.getsize(os.path.join(ARTIFACTS_DIR, f))
    print(f"  {f} ({size:,} bytes)")
