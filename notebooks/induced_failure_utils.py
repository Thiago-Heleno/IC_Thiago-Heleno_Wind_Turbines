"""
Shared utilities for Induced Failure Notebooks (NB4 & NB5).
Standalone from existing pipelines — reuses no code from NB1/NB2/NB3.

Contents:
  - CARE data loading & schema normalization
  - Preprocessing (clipping, imputation, scaling)
  - Unsupervised feature selection
  - Synthetic failure injection (Gaussian, Drift, Spike)
  - CARE Score evaluation
"""

import os
import gc
import json
import random
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, roc_curve, average_precision_score,
    roc_auc_score, fbeta_score, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score
)
from pathlib import Path

SEED = 42

# ─────────────────────────────────────────────
# 1. Project & Data Resolution
# ─────────────────────────────────────────────


def resolve_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        if (base / "README.md").exists() and (base / "notebooks").exists():
            return base
        if (base / "CARE_To_Compare").is_dir():
            return base
    return cwd


def resolve_care_paths(project_root: Path) -> Tuple[str, str, str]:
    env = os.environ.get("CARE_WIND_FARM_C", "").strip()
    if env and os.path.isdir(os.path.join(env, "datasets")):
        base = os.path.normpath(env)
    else:
        base = str(project_root / "CARE_To_Compare" / "Wind Farm C")
    return (
        os.path.join(base, "datasets"),
        os.path.join(base, "event_info.csv"),
        os.path.join(base, "feature_description.csv"),
    )


# ─────────────────────────────────────────────
# 2. CARE CSV Loading
# ─────────────────────────────────────────────

META_COLS = {
    "timestamp", "status_id", "train_test", "asset_id", "source_file",
    "event_id", "event_label", "is_padding", "id",
    "time_stamp", "status_type_id",
}


def _detect_sep(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        head = f.readline()
    return ";" if head.count(";") > head.count(",") else ","


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        c0 = str(c).strip().lstrip("\ufeff")
        if c0 != c:
            rename[c] = c0
    if rename:
        df = df.rename(columns=rename)
    lower = {str(x).lower().replace(" ", "_"): x for x in df.columns}
    if "timestamp" not in df.columns:
        for key in ("time_stamp", "timestamp", "datetime"):
            if key in lower:
                df = df.rename(columns={lower[key]: "timestamp"})
                break
    if "status_id" not in df.columns and "status_type_id" in lower:
        df = df.rename(columns={lower["status_type_id"]: "status_id"})
    return df


def load_care_csvs(datasets_dir: str) -> pd.DataFrame:
    """Load all Wind Farm C CSVs into a single DataFrame."""
    csv_files = sorted([f for f in os.listdir(datasets_dir) if f.endswith(".csv")])
    if not csv_files:
        raise FileNotFoundError(f"No CSVs in {datasets_dir}")

    frames = []
    for f in csv_files:
        path = os.path.join(datasets_dir, f)
        sep = _detect_sep(path)
        df = pd.read_csv(path, sep=sep, low_memory=True)
        # Downcast floats
        for c in df.select_dtypes(include=[np.number]).columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].astype(np.float32)
        df["source_file"] = f
        frames.append(df)
        if len(frames) % 10 == 0:
            gc.collect()

    df_all = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    df_all = _normalize_schema(df_all)
    if "timestamp" in df_all.columns:
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
    return df_all.sort_values("timestamp").reset_index(drop=True)


def load_event_info(path: str) -> pd.DataFrame:
    sep = _detect_sep(path)
    return pd.read_csv(path, sep=sep)


def load_feature_desc(path: str) -> pd.DataFrame:
    sep = _detect_sep(path)
    return pd.read_csv(path, sep=sep)


def merge_event_labels(df: pd.DataFrame, event_info: pd.DataFrame) -> pd.DataFrame:
    """Add event_label column based on source_file → event_id mapping."""
    emap = {}
    for _, row in event_info.iterrows():
        eid = row["event_id"]
        emap[f"{eid}.csv"] = row["event_label"]
    df["event_label"] = df["source_file"].map(emap).fillna("unknown")
    return df


def infer_binary_labels(df: pd.DataFrame) -> np.ndarray:
    if "event_label" in df.columns:
        return (~df["event_label"].astype(str).str.lower().eq("normal")).astype(int).values
    return df["status_id"].isin([3, 4]).astype(int).values


# ─────────────────────────────────────────────
# 3. Preprocessing
# ─────────────────────────────────────────────

def infer_feature_columns(df: pd.DataFrame) -> List[str]:
    meta = META_COLS | {"label", "source_file", "event_label",
                        "event_start", "event_end", "event_start_id",
                        "event_end_id", "event_description", "event_id"}
    return [c for c in df.columns
            if c not in meta and pd.api.types.is_numeric_dtype(df[c])]


class DataClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_q=0.001, upper_q=0.999):
        self.lower_q, self.upper_q = lower_q, upper_q
        self.lo_, self.hi_ = None, None

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=np.float32)
        self.lo_ = np.nanquantile(arr, self.lower_q, axis=0)
        self.hi_ = np.nanquantile(arr, self.upper_q, axis=0)
        return self

    def transform(self, X):
        arr = np.array(X, dtype=np.float32)
        return np.clip(arr, self.lo_, self.hi_)


class NanImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fill_values_ = None

    def fit(self, X, y=None):
        self.fill_values_ = np.nanmean(np.asarray(X, dtype=np.float32), axis=0)
        self.fill_values_ = np.nan_to_num(self.fill_values_, 0.0)
        return self

    def transform(self, X):
        arr = np.array(X, dtype=np.float32)
        for j in range(arr.shape[1]):
            mask = np.isnan(arr[:, j])
            if mask.any():
                arr[mask, j] = self.fill_values_[j]
        return arr


def build_preprocessor() -> Pipeline:
    return Pipeline([
        ("clipper", DataClipper()),
        ("imputer", NanImputer()),
        ("scaler", StandardScaler()),
    ])


# ─────────────────────────────────────────────
# 4. Feature Selection (unsupervised)
# ─────────────────────────────────────────────

def select_features_unsupervised(
    X_train: np.ndarray,
    feature_names: List[str],
    var_threshold: float = 1e-4,
    corr_threshold: float = 0.95,
) -> Tuple[List[int], List[str]]:
    """3-stage unsupervised feature selection. Returns selected indices and names."""

    # Stage 1: Remove low variance
    variances = np.var(X_train, axis=0)
    keep_mask = variances > var_threshold
    indices = np.where(keep_mask)[0].tolist()

    # Stage 2: Remove highly correlated features
    if len(indices) > 1:
        sub = X_train[:, indices]
        # Sample for speed
        n_sample = min(50000, len(sub))
        if n_sample < len(sub):
            idx = np.random.choice(len(sub), n_sample, replace=False)
            sub = sub[idx]
        corr = np.corrcoef(sub, rowvar=False)
        corr = np.nan_to_num(corr, 0.0)
        drop = set()
        for i in range(len(corr)):
            if i in drop:
                continue
            for j in range(i + 1, len(corr)):
                if j in drop:
                    continue
                if abs(corr[i, j]) > corr_threshold:
                    drop.add(j)
        indices = [indices[i] for i in range(len(indices)) if i not in drop]

    names = [feature_names[i] for i in indices]
    return indices, names


# ─────────────────────────────────────────────
# 5. Synthetic Failure Injection
# ─────────────────────────────────────────────

PHYSICAL_KEYS = [
    "temp", "vibr", "speed", "power", "press", "bearing",
    "rotor", "generator", "gear", "hydraul", "oil", "wind",
]


def identify_physical_features(feature_names: List[str]) -> List[int]:
    """Return indices of physically meaningful features."""
    indices = []
    for i, name in enumerate(feature_names):
        low = name.lower()
        if any(k in low for k in PHYSICAL_KEYS):
            indices.append(i)
    # Fallback: if too few, use all
    if len(indices) < 5:
        indices = list(range(len(feature_names)))
    return indices


def inject_gaussian_noise(
    X: np.ndarray,
    target_indices: List[int],
    sigma_multiplier: float = 3.0,
    block_size: int = 12,
    rng: np.random.RandomState = None,
) -> np.ndarray:
    """Add Gaussian noise to a contiguous block of samples on target features."""
    X_out = X.copy()
    n = len(X_out)
    if n < block_size:
        return X_out
    start = rng.randint(0, max(1, n - block_size))
    end = min(start + block_size, n)
    stds = np.std(X_out[:, target_indices], axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    noise = rng.randn(end - start, len(target_indices)) * sigma_multiplier * stds
    X_out[start:end][:, target_indices] += noise.astype(np.float32)
    return X_out


def inject_drift(
    X: np.ndarray,
    target_indices: List[int],
    magnitude_multiplier: float = 5.0,
    rng: np.random.RandomState = None,
) -> np.ndarray:
    """Inject gradual linear drift on target features."""
    X_out = X.copy()
    n = len(X_out)
    n_drift = min(n, max(6, n // 2))
    start = rng.randint(0, max(1, n - n_drift))
    end = start + n_drift
    stds = np.std(X_out[:, target_indices], axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    ramp = np.linspace(0, 1, n_drift).reshape(-1, 1)
    direction = rng.choice([-1, 1], size=(1, len(target_indices)))
    drift = ramp * magnitude_multiplier * stds * direction
    X_out[start:end][:, target_indices] += drift.astype(np.float32)
    return X_out


def inject_spikes(
    X: np.ndarray,
    target_indices: List[int],
    magnitude_multiplier: float = 8.0,
    n_spikes: int = 3,
    rng: np.random.RandomState = None,
) -> np.ndarray:
    """Inject sudden spikes on random target features."""
    X_out = X.copy()
    n = len(X_out)
    stds = np.std(X_out[:, target_indices], axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    for _ in range(min(n_spikes, n)):
        row = rng.randint(0, n)
        feat = rng.randint(0, len(target_indices))
        direction = rng.choice([-1, 1])
        X_out[row, target_indices[feat]] += direction * magnitude_multiplier * stds[feat]
    return X_out


def inject_synthetic_failures(
    X_normal: np.ndarray,
    feature_names: List[str],
    injection_fraction: float = 0.07,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inject synthetic failures into normal data.

    Returns:
        X_combined: original normal + induced anomaly samples
        y_combined: labels (0=normal, 2=induced)
        injection_types: array of strings ('none','gaussian','drift','spike')
    """
    rng = np.random.RandomState(seed)
    phys_idx = identify_physical_features(feature_names)

    n_total = len(X_normal)
    n_inject = int(n_total * injection_fraction)
    n_per_type = max(1, n_inject // 3)

    # Select random normal samples to corrupt
    all_inject_idx = rng.choice(n_total, min(n_per_type * 3, n_total), replace=False)

    X_induced = []
    types = []

    # Gaussian
    for i in range(min(n_per_type, len(all_inject_idx))):
        idx = all_inject_idx[i]
        # Take a small window around the sample for context
        start = max(0, idx - 6)
        end = min(n_total, idx + 6)
        block = X_normal[start:end].copy()
        # Select subset of physical features
        n_feat = rng.randint(3, min(len(phys_idx), 10) + 1)
        feat_subset = list(rng.choice(phys_idx, n_feat, replace=False))
        block = inject_gaussian_noise(block, feat_subset, sigma_multiplier=3.0, rng=rng)
        # Take middle sample
        mid = min(idx - start, len(block) - 1)
        X_induced.append(block[mid])
        types.append("gaussian")

    # Drift
    offset = n_per_type
    for i in range(min(n_per_type, len(all_inject_idx) - offset)):
        idx = all_inject_idx[offset + i]
        start = max(0, idx - 6)
        end = min(n_total, idx + 6)
        block = X_normal[start:end].copy()
        n_feat = rng.randint(2, min(len(phys_idx), 8) + 1)
        feat_subset = list(rng.choice(phys_idx, n_feat, replace=False))
        block = inject_drift(block, feat_subset, magnitude_multiplier=5.0, rng=rng)
        mid = min(idx - start, len(block) - 1)
        X_induced.append(block[mid])
        types.append("drift")

    # Spikes
    offset = 2 * n_per_type
    for i in range(min(n_per_type, len(all_inject_idx) - offset)):
        idx = all_inject_idx[offset + i]
        sample = X_normal[idx].copy().reshape(1, -1)
        n_feat = rng.randint(1, min(len(phys_idx), 5) + 1)
        feat_subset = list(rng.choice(phys_idx, n_feat, replace=False))
        sample = inject_spikes(sample, feat_subset, magnitude_multiplier=8.0, n_spikes=n_feat, rng=rng)
        X_induced.append(sample[0])
        types.append("spike")

    X_induced = np.array(X_induced, dtype=np.float32)

    # Combine: normal + induced
    X_combined = np.concatenate([X_normal, X_induced], axis=0)
    y_combined = np.concatenate([
        np.zeros(n_total, dtype=np.int32),
        np.full(len(X_induced), 2, dtype=np.int32),  # 2 = induced
    ])
    type_arr = np.array(["none"] * n_total + types)

    # Shuffle
    perm = rng.permutation(len(X_combined))
    return X_combined[perm], y_combined[perm], type_arr[perm]


# ─────────────────────────────────────────────
# 6. CARE Score Evaluation
# ─────────────────────────────────────────────

def compute_criticality(alarm_flags: np.ndarray, threshold: int = 6) -> Tuple[np.ndarray, bool]:
    crit = np.zeros(len(alarm_flags), dtype=int)
    for i in range(1, len(alarm_flags)):
        if alarm_flags[i]:
            crit[i] = crit[i - 1] + 1
        else:
            crit[i] = max(crit[i - 1] - 1, 0)
    return crit, bool(crit.max() >= threshold)


def weighted_score_earliness(event_alarm: np.ndarray) -> float:
    m = len(event_alarm)
    if m == 0:
        return 0.0
    weights = np.ones(m)
    half = m / 2.0
    for i in range(m):
        if i > half:
            weights[i] = max(0.0, 1.0 - (i - half) / half)
    denom = weights.sum()
    return float((weights * event_alarm).sum() / denom) if denom > 0 else 0.0


def fbeta_safe(y_true, y_pred, beta=0.5):
    if len(np.unique(y_true)) <= 1:
        return 0.0
    return float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))


def care_score_formula(fbeta_mean, acc_mean, efbeta, ws_mean, any_alarm):
    if not any_alarm:
        return 0.0
    if acc_mean < 0.5:
        return float(acc_mean)
    return float((fbeta_mean + ws_mean + efbeta + 2.0 * acc_mean) / 5.0)


def evaluate_care_per_dataset(
    df_eval: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    criticality_threshold: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate CARE score per dataset (source_file).

    Returns:
        care_by_dataset: per-dataset metrics
        care_summary: single-row summary
    """
    df_eval = df_eval.copy()
    df_eval["y_true"] = y_true
    df_eval["y_pred"] = y_pred
    df_eval["score"] = scores

    dataset_rows = []
    y_event_true, y_event_pred = [], []

    for src, g in df_eval.groupby("source_file", sort=False):
        gt = g["y_true"].values.astype(int)
        gp = g["y_pred"].values.astype(int)
        has_anomaly = bool((gt == 1).any())

        f12 = fbeta_safe(gt, gp, beta=0.5) if len(np.unique(gt)) > 1 else 0.0
        normal_mask = gt == 0
        acc = float((gp[normal_mask] == 0).sum() / max(1, normal_mask.sum())) if normal_mask.any() else 1.0

        crit, alarm = compute_criticality(gp.astype(bool), threshold=criticality_threshold)

        anom_mask = gt == 1
        ws = weighted_score_earliness(gp[anom_mask]) if has_anomaly and anom_mask.any() else np.nan

        pr_auc, roc_auc = np.nan, np.nan
        if len(np.unique(gt)) > 1:
            try:
                pr_auc = float(average_precision_score(gt, g["score"].values))
                roc_auc = float(roc_auc_score(gt, g["score"].values))
            except Exception:
                pass

        dataset_rows.append({
            "source_file": src, "has_anomaly": int(has_anomaly),
            "alarm": int(alarm), "F1_2_dataset": f12, "Acc_dataset": acc,
            "WS_dataset": ws, "criticality_max": int(crit.max()),
            "PR_AUC": pr_auc, "ROC_AUC": roc_auc,
        })
        y_event_true.append(int(has_anomaly))
        y_event_pred.append(int(alarm))

    care_ds = pd.DataFrame(dataset_rows)
    anom_ds = care_ds["has_anomaly"] == 1
    norm_ds = care_ds["has_anomaly"] == 0

    f12_mean = float(care_ds.loc[anom_ds, "F1_2_dataset"].mean()) if anom_ds.any() else 0.0
    ws_mean = float(care_ds.loc[anom_ds, "WS_dataset"].dropna().mean()) if anom_ds.any() else 0.0
    acc_mean = float(care_ds.loc[norm_ds, "Acc_dataset"].mean()) if norm_ds.any() else 0.0
    efbeta = fbeta_safe(np.array(y_event_true), np.array(y_event_pred), beta=0.5)
    any_alarm = bool(np.any(np.array(y_event_pred) == 1))
    care = care_score_formula(f12_mean, acc_mean, efbeta, ws_mean, any_alarm)

    summary = pd.DataFrame([{
        "F1_2": f12_mean, "Acc": acc_mean, "EF1_2": efbeta,
        "WS": ws_mean, "CARE": care,
        "n_datasets": len(care_ds),
        "n_anomalous": int(anom_ds.sum()),
        "n_normal": int(norm_ds.sum()),
    }])
    return care_ds, summary


def print_classification_report(y_true, y_pred, scores, title=""):
    """Print standard binary classification metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    print(f"  Precision:  {p:.4f}")
    print(f"  Recall:     {r:.4f}")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"  Accuracy:   {acc:.4f}")
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, scores)
        print(f"  AUC-ROC:    {auc:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:\n{cm}")
    return {"precision": p, "recall": r, "f1": f1, "accuracy": acc}
