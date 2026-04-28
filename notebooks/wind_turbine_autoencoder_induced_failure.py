# === Cell 1: ConfiguraÃ§Ã£o e Imports ===
import os, sys, gc, json, warnings, random
import numpy as np
import pandas as pd

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
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.abspath("__file__")))
from induced_failure_utils import *

PROJECT_ROOT = resolve_project_root()
DATASETS_DIR, EVENT_INFO_PATH, FEATURE_DESC_PATH = resolve_care_paths(PROJECT_ROOT)
ARTIFACTS_DIR = str(PROJECT_ROOT / "resultados" / "05_autoencoder_induced_failure")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

N_OPTUNA_TRIALS = int(os.environ.get("N_OPTUNA_TRIALS", "30"))
N_OPTUNA_TRIALS_GAMMA = int(os.environ.get("N_OPTUNA_TRIALS_GAMMA", "30"))
INJECTION_FRACTION = 0.07

print(f"TensorFlow: {tf.__version__}")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATASETS_DIR: {DATASETS_DIR}")

# === Cell 2: IngestÃ£o de Dados e Split Temporal ===

df_raw = load_care_csvs(DATASETS_DIR)
event_info = load_event_info(EVENT_INFO_PATH)
df_raw = merge_event_labels(df_raw, event_info)

FEATURE_COLS = infer_feature_columns(df_raw)
print(f"Features numÃ©ricas: {len(FEATURE_COLS)}")
print(f"Total amostras: {len(df_raw):,}")

# Use CARE train_test column for temporal split
train_mask = df_raw["train_test"].astype(str).str.lower().eq("train")
test_mask = df_raw["train_test"].astype(str).str.lower().isin(["test", "prediction"])
normal_mask = ~df_raw["status_id"].isin([3, 4])

df_train_all = df_raw.loc[train_mask].copy()
df_test = df_raw.loc[test_mask].copy()

# Sub-split: train normal â†’ AE training (80%) + calibration (20%)
df_train_normal = df_train_all.loc[train_mask & normal_mask].copy()
df_sorted = df_train_normal.sort_values("timestamp").reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.80)
df_train_ae = df_sorted.iloc[:split_idx].copy()
df_cal = df_sorted.iloc[split_idx:].copy()

del df_raw, df_train_normal, df_sorted; gc.collect()

print(f"Train AE (normal only): {len(df_train_ae):,}")
print(f"Calibration (normal): {len(df_cal):,}")
print(f"Test (all): {len(df_test):,}")

y_test = infer_binary_labels(df_test)
print(f"Test anomalies: {y_test.sum():,} / {len(y_test):,} ({100*y_test.mean():.2f}%)")

# === Cell 3: PrÃ©-processamento + SeleÃ§Ã£o de Features ===

X_train_raw = df_train_ae[FEATURE_COLS].values.astype(np.float32)
X_cal_raw = df_cal[FEATURE_COLS].values.astype(np.float32)
X_test_raw = df_test[FEATURE_COLS].values.astype(np.float32)

# Fit preprocessor on TRAIN NORMAL only (which is all of df_train_ae)
preprocessor = build_preprocessor()
preprocessor.fit(X_train_raw)

X_train_proc = preprocessor.transform(X_train_raw).astype(np.float32)
X_cal_proc = preprocessor.transform(X_cal_raw).astype(np.float32)
X_test_proc = preprocessor.transform(X_test_raw).astype(np.float32)
del X_train_raw, X_cal_raw, X_test_raw; gc.collect()

# Feature selection
sel_indices, sel_names = select_features_unsupervised(X_train_proc, FEATURE_COLS)
print(f"Features selecionadas: {len(sel_indices)} de {len(FEATURE_COLS)}")

X_train = X_train_proc[:, sel_indices]
X_cal = X_cal_proc[:, sel_indices]
X_test = X_test_proc[:, sel_indices]
del X_train_proc, X_cal_proc, X_test_proc; gc.collect()

INPUT_DIM = X_train.shape[1]
print(f"Shapes: Train={X_train.shape}, Cal={X_cal.shape}, Test={X_test.shape}")

# === Cell 4: Optuna â€” OtimizaÃ§Ã£o do Autoencoder ===

def build_autoencoder(input_dim, n_layers, code_size, learning_rate, decay_rate):
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    # Encoder
    layer_sizes = [200, 100, 50][:n_layers]
    for units in layer_sizes:
        x = layers.Dense(units, kernel_initializer="he_normal")(x)
        x = layers.PReLU()(x)
    x = layers.Dense(code_size, kernel_initializer="he_normal", name="code")(x)
    x = layers.PReLU()(x)
    # Decoder (mirror)
    for units in reversed(layer_sizes):
        x = layers.Dense(units, kernel_initializer="he_normal")(x)
        x = layers.PReLU()(x)
    outputs = layers.Dense(input_dim, activation="linear", name="recon")(x)

    model = keras.Model(inputs, outputs)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate, decay_steps=1000,
        decay_rate=decay_rate, staircase=False,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss="mse", metrics=["mae"])
    return model

def rmse_per_sample(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))

def objective_ae(trial):
    keras.backend.clear_session(); gc.collect()
    n_layers = trial.suggest_int("n_layers", 1, 3)
    code_size = trial.suggest_int("code_size", 10, 64)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    decay = trial.suggest_float("decay_rate", 0.90, 0.999)

    # Internal split for Optuna
    n_split = int(len(X_train) * 0.80)
    Xtr, Xvl = X_train[:n_split], X_train[n_split:]

    model = build_autoencoder(INPUT_DIM, n_layers, code_size, lr, decay)
    model.fit(Xtr, Xtr, validation_data=(Xvl, Xvl), epochs=30,
              batch_size=128, shuffle=False, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=5,
                         restore_best_weights=True)])
    val_loss = model.evaluate(Xvl, Xvl, verbose=0)[0]
    del model; keras.backend.clear_session(); gc.collect()
    return float(val_loss)

sampler = TPESampler(seed=SEED)
study = optuna.create_study(direction="minimize", sampler=sampler, study_name="ae_induced")
study.optimize(objective_ae, n_trials=N_OPTUNA_TRIALS)

best_params = study.best_trial.params
print(f"Best params: {best_params}")
print(f"Best val_loss: {study.best_value:.6f}")

# === Cell 5: Treinamento Final do Autoencoder ===

keras.backend.clear_session(); gc.collect()

n_split = int(len(X_train) * 0.80)
Xtr, Xvl = X_train[:n_split], X_train[n_split:]

final_ae = build_autoencoder(
    INPUT_DIM,
    n_layers=best_params["n_layers"],
    code_size=best_params["code_size"],
    learning_rate=best_params["learning_rate"],
    decay_rate=best_params["decay_rate"],
)

CKPT_PATH = os.path.join(ARTIFACTS_DIR, "autoencoder_best.keras")
history = final_ae.fit(
    Xtr, Xtr, validation_data=(Xvl, Xvl),
    epochs=200, batch_size=128, shuffle=False, verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping("val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(CKPT_PATH, "val_loss", save_best_only=True, verbose=1),
    ],
)

# Compute RMSE for all splits
recon_train = final_ae.predict(X_train, verbose=0)
recon_cal = final_ae.predict(X_cal, verbose=0)
recon_test = final_ae.predict(X_test, verbose=0)

rmse_train = rmse_per_sample(X_train, recon_train)
rmse_cal = rmse_per_sample(X_cal, recon_cal)
rmse_test = rmse_per_sample(X_test, recon_test)

del recon_train, recon_cal, recon_test; gc.collect()
print(f"RMSE train: mean={rmse_train.mean():.4f}, std={rmse_train.std():.4f}")
print(f"RMSE cal:   mean={rmse_cal.mean():.4f}, std={rmse_cal.std():.4f}")
print(f"RMSE test:  mean={rmse_test.mean():.4f}, std={rmse_test.std():.4f}")

# === Cell 6: InjeÃ§Ã£o de Falhas SintÃ©ticas na CalibraÃ§Ã£o ===

# Inject induced anomalies into calibration normal data
X_cal_aug, y_cal_aug, types_cal = inject_synthetic_failures(
    X_cal,
    feature_names=sel_names,
    injection_fraction=INJECTION_FRACTION,
    seed=SEED + 1,  # different seed from classifier
)

# Compute RMSE for augmented calibration
recon_cal_aug = final_ae.predict(X_cal_aug, verbose=0)
rmse_cal_aug = rmse_per_sample(X_cal_aug, recon_cal_aug)
del recon_cal_aug; gc.collect()

# Separate RMSE by class
rmse_normal_cal = rmse_cal_aug[y_cal_aug == 0]
rmse_induced_cal = rmse_cal_aug[y_cal_aug == 2]

print(f"Calibration RMSE (normal):  mean={rmse_normal_cal.mean():.4f}, P95={np.percentile(rmse_normal_cal, 95):.4f}")
print(f"Calibration RMSE (induced): mean={rmse_induced_cal.mean():.4f}, P95={np.percentile(rmse_induced_cal, 95):.4f}")

# Get RMSE for real anomalies in test (for reference)
rmse_test_normal = rmse_test[y_test == 0]
rmse_test_anomaly = rmse_test[y_test == 1]
print(f"\nTest RMSE (normal):  mean={rmse_test_normal.mean():.4f}")
print(f"Test RMSE (anomaly): mean={rmse_test_anomaly.mean():.4f}")
print("\nExpected ordering: Normal < Induced < Real Anomaly")
print(f"  {rmse_normal_cal.mean():.4f} < {rmse_induced_cal.mean():.4f} ? {'YES' if rmse_induced_cal.mean() > rmse_normal_cal.mean() else 'NO'}")

# === Cell 7: CalibraÃ§Ã£o de Threshold â€” PadrÃ£o vs Induzido ===

# --- Standard threshold (P95 of normal calibration RMSE) ---
threshold_p95 = float(np.percentile(rmse_normal_cal, 95))
print(f"Standard threshold (P95 normal): {threshold_p95:.4f}")

# --- Induced-calibrated threshold ---
# Key insight: set threshold above the P95 of INDUCED anomaly RMSE
# This ensures induced noise does NOT trigger alarms â†’ fewer FP
threshold_induced_p95 = float(np.percentile(rmse_induced_cal, 95))
print(f"Induced threshold (P95 induced): {threshold_induced_p95:.4f}")

# --- Optuna gamma search for adaptive threshold ---
# threshold = base + gamma, where base = P50 of induced RMSE
base_induced = float(np.percentile(rmse_induced_cal, 50))

def objective_gamma(trial):
    gamma = trial.suggest_float("gamma", 0.0, 2.0)
    threshold = base_induced + gamma
    # Evaluate on calibration: we want to reject induced but detect anomaly-like patterns
    pred_normal = (rmse_normal_cal > threshold).astype(int)
    fpr = pred_normal.mean()  # False positive rate on normal data
    # We want FPR < 5%
    score = max(0, 1.0 - 20 * max(0, fpr - 0.05))  # Penalize FPR > 5%
    return float(score)

gamma_study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
gamma_study.optimize(objective_gamma, n_trials=N_OPTUNA_TRIALS_GAMMA)
gamma_best = gamma_study.best_trial.params["gamma"]
threshold_adaptive = base_induced + gamma_best

print(f"\nGamma best: {gamma_best:.4f}")
print(f"Adaptive threshold (induced_P50 + gamma): {threshold_adaptive:.4f}")
print("\nThreshold comparison:")
print(f"  Standard (P95 normal):    {threshold_p95:.4f}")
print(f"  Induced (P95 induced):    {threshold_induced_p95:.4f}")
print(f"  Adaptive (P50_ind+gamma): {threshold_adaptive:.4f}")

# === Cell 8: AvaliaÃ§Ã£o no Teste + CARE Score ===

thresholds_to_test = {
    "Standard (P95)": threshold_p95,
    "Induced (P95 induced)": threshold_induced_p95,
    "Adaptive (P50_ind+gamma)": threshold_adaptive,
}

all_results = {}
all_care = {}

for name, th in thresholds_to_test.items():
    y_pred = (rmse_test > th).astype(int)
    metrics = print_classification_report(y_test, y_pred, rmse_test, title=f"{name} â€” threshold={th:.4f}")
    all_results[name] = {**metrics, "threshold": th}

    # CARE Score
    care_ds, care_summary = evaluate_care_per_dataset(
        df_test, y_test, y_pred, rmse_test
    )
    all_care[name] = {"ds": care_ds, "summary": care_summary}
    print(f"  CARE Score: {float(care_summary['CARE'].iloc[0]):.4f}")

# Comparison table
comp_rows = []
for name, m in all_results.items():
    care_val = float(all_care[name]["summary"]["CARE"].iloc[0])
    comp_rows.append({"Threshold": name, **m, "CARE": care_val})
comp = pd.DataFrame(comp_rows)
print("\n" + "="*70)
print("  COMPARISON â€” All Thresholds")
print("="*70)
print(comp.to_string(index=False))

# === Cell 9: VisualizaÃ§Ãµes ===

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. AE Training loss
axes[0,0].plot(history.history["loss"], label="Train")
axes[0,0].plot(history.history["val_loss"], label="Val")
axes[0,0].set_title("AE Loss"); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

# 2. RMSE Distribution: Normal vs Induced vs Real Anomaly
axes[0,1].hist(rmse_normal_cal, bins=80, alpha=0.6, label="Normal (cal)", density=True, color="#4C72B0")
axes[0,1].hist(rmse_induced_cal, bins=80, alpha=0.6, label="Induced (cal)", density=True, color="#FFA500")
axes[0,1].hist(rmse_test_anomaly, bins=80, alpha=0.6, label="Real Anomaly (test)", density=True, color="#C44E52")
axes[0,1].axvline(threshold_p95, color="blue", linestyle="--", label=f"P95 std={threshold_p95:.3f}")
axes[0,1].axvline(threshold_induced_p95, color="orange", linestyle="--", label=f"P95 ind={threshold_induced_p95:.3f}")
axes[0,1].set_title("RMSE Distribution (3-class)"); axes[0,1].legend(fontsize=7); axes[0,1].grid(alpha=0.3)

# 3. RMSE time series (test, first 2000 samples)
n_show = min(2000, len(rmse_test))
axes[0,2].plot(rmse_test[:n_show], alpha=0.7, linewidth=0.5, label="RMSE")
axes[0,2].axhline(threshold_p95, color="blue", linestyle="--", alpha=0.7, label="Std threshold")
axes[0,2].axhline(threshold_induced_p95, color="orange", linestyle="--", alpha=0.7, label="Induced threshold")
anom_idx = np.where(y_test[:n_show] == 1)[0]
if len(anom_idx) > 0:
    axes[0,2].scatter(anom_idx, rmse_test[anom_idx], s=3, color="red", alpha=0.5, label="True anomaly")
axes[0,2].set_title("RMSE Time Series (test)"); axes[0,2].legend(fontsize=7); axes[0,2].grid(alpha=0.3)

# 4-5. Confusion matrices for standard vs induced threshold
for ax_idx, (name, th) in enumerate(list(thresholds_to_test.items())[:2]):
    y_pred_cm = (rmse_test > th).astype(int)
    cm = confusion_matrix(y_test, y_pred_cm)
    cmap = "Blues" if ax_idx == 0 else "Oranges"
    axes[1, ax_idx].imshow(cm, cmap=cmap)
    for i in range(2):
        for j in range(2):
            axes[1, ax_idx].text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=11)
    axes[1, ax_idx].set_title(f"CM: {name}")
    axes[1, ax_idx].set_xlabel("Predicted"); axes[1, ax_idx].set_ylabel("True")
    axes[1, ax_idx].set_xticks([0,1]); axes[1, ax_idx].set_yticks([0,1])
    axes[1, ax_idx].set_xticklabels(["Normal","Anomaly"]); axes[1, ax_idx].set_yticklabels(["Normal","Anomaly"])

# 6. CARE sub-scores comparison
subscores = ["F1_2", "Acc", "EF1_2", "WS", "CARE"]
x = np.arange(len(subscores))
w = 0.25
for i, (name, data) in enumerate(all_care.items()):
    vals = [float(data["summary"][s].iloc[0]) for s in subscores]
    axes[1,2].bar(x + i*w, vals, w, label=name[:15])
axes[1,2].set_xticks(x + w); axes[1,2].set_xticklabels(subscores)
axes[1,2].set_ylim(0, 1); axes[1,2].set_title("CARE Sub-scores"); axes[1,2].legend(fontsize=6)

plt.suptitle("Autoencoder â€” Standard vs Induced Threshold Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS_DIR, "comparison_plots.png"), dpi=150, bbox_inches="tight")
plt.show()

# === Cell 10: ExportaÃ§Ã£o de Artefatos ===

final_ae.save(os.path.join(ARTIFACTS_DIR, "autoencoder_induced.keras"))

# Best threshold (pick the one with best CARE)
best_th_name = max(all_results, key=lambda k: all_care[k]["summary"]["CARE"].iloc[0])
best_th_val = all_results[best_th_name]["threshold"]

all_metrics = {
    "notebook": "wind_turbine_autoencoder_induced_failure",
    "paradigma": "semi-supervisionado",
    "modelo": "MLP Autoencoder + Induced Failure Calibration",
    "dataset": "CARE_To_Compare Wind Farm C",
    "hiperparametros": best_params,
    "thresholds": {name: m["threshold"] for name, m in all_results.items()},
    "best_threshold": {"name": best_th_name, "value": best_th_val},
    "injection_fraction": INJECTION_FRACTION,
    "gamma": gamma_best,
    "metricas_teste": {name: {k: v for k, v in m.items() if k != "threshold"}
                       for name, m in all_results.items()},
    "metricas_CARE": {name: all_care[name]["summary"].iloc[0].to_dict()
                       for name in all_care},
}
with open(os.path.join(ARTIFACTS_DIR, "metricas.json"), "w") as f:
    json.dump(all_metrics, f, indent=2, default=str)

# Save CARE per-dataset (best threshold)
all_care[best_th_name]["ds"].to_csv(os.path.join(ARTIFACTS_DIR, "care_results.csv"), index=False)
all_care[best_th_name]["summary"].to_csv(os.path.join(ARTIFACTS_DIR, "care_summary.csv"), index=False)

with open(os.path.join(ARTIFACTS_DIR, "best_params.json"), "w") as f:
    json.dump(best_params, f, indent=2)

comp.to_csv(os.path.join(ARTIFACTS_DIR, "comparison.csv"), index=False)

with open(os.path.join(ARTIFACTS_DIR, "threshold_params.json"), "w") as f:
    json.dump({
        "standard_p95": threshold_p95,
        "induced_p95": threshold_induced_p95,
        "adaptive": threshold_adaptive,
        "gamma": gamma_best,
        "base_induced_p50": base_induced,
        "best_threshold_name": best_th_name,
    }, f, indent=2)

print(f"Artefatos exportados em: {ARTIFACTS_DIR}")
for f_name in sorted(os.listdir(ARTIFACTS_DIR)):
    size = os.path.getsize(os.path.join(ARTIFACTS_DIR, f_name))
    print(f"  {f_name} ({size:,} bytes)")
