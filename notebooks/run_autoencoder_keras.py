# === Cell 1 ===
import os

# CPU Limited to 4-6 cores
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

# Isolate GPU to the idle one (1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Framework-Specific Memory Management (Keras/TensorFlow)
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# === Cell 2 ===
# Etapa 0 - Configuracao e dependencias
import os

# --- Laboratorio (maquina compartilhada): ANTES de importar TensorFlow ---
_lab_threads = os.environ.get("LAB_CPU_THREADS", "6").strip()
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, _lab_threads)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("LAB_CUDA_DEVICE", "1")

import json
import math
import random
import pickle
import gc
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, auc, fbeta_score, roc_auc_score, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import optuna
from optuna.samplers import TPESampler
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.threading.set_intra_op_parallelism_threads(max(1, int(_lab_threads)))
    tf.config.threading.set_inter_op_parallelism_threads(1)
except (ValueError, TypeError):
    pass


def _resolve_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        if (base / "README.md").exists() and (base / "notebooks").exists():
            return base
        if (base / "CARE_To_Compare").is_dir():
            return base
    return cwd


PROJECT_ROOT = _resolve_project_root()
_env_care = os.environ.get("CARE_WIND_FARM_C", "").strip()
if _env_care and os.path.isdir(os.path.join(_env_care, "datasets")):
    BASE_DIR = os.path.normpath(_env_care)
else:
    BASE_DIR = os.path.join(str(PROJECT_ROOT), "CARE_To_Compare", "Wind Farm C")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
EVENT_INFO_PATH = os.path.join(BASE_DIR, "event_info.csv")
FEATURE_DESC_PATH = os.path.join(BASE_DIR, "feature_description.csv")

ARTIFACTS_DIR = os.path.join(str(PROJECT_ROOT), "resultados", "results_keras_pipeline")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

N_OPTUNA_TRIALS_AE = int(os.environ.get("N_OPTUNA_TRIALS_AE", "50"))
N_OPTUNA_TRIALS_GAMMA = int(os.environ.get("N_OPTUNA_TRIALS_GAMMA", "50"))

print("TensorFlow:", tf.__version__)
print("Optuna:", optuna.__version__)
print(
    "Recursos (lab): CUDA_VISIBLE_DEVICES=%s  LAB_CPU_THREADS=%s"
    % (os.environ.get("CUDA_VISIBLE_DEVICES", ""), _lab_threads)
)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATASETS_DIR:", DATASETS_DIR)


# === Cell 3 ===
# Etapa 1 - Ingestao dos CSVs wide e split temporal
REQUIRED_COLS = ["timestamp", "status_id", "train_test", "asset_id"]


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Reduz RAM sem fragmentar o bloco: um unico astype por tipos."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return df
    casts = {}
    for c in num_cols:
        s = df[c]
        if pd.api.types.is_float_dtype(s):
            casts[c] = np.float32
        elif pd.api.types.is_integer_dtype(s):
            casts[c] = pd.to_numeric(s, downcast="integer").dtype
    return df.astype(casts, copy=False)


def normalize_care_wide_schema(df: pd.DataFrame) -> pd.DataFrame:
    """CARE Wind Farm C usa time_stamp e status_type_id; o pipeline espera timestamp e status_id."""
    rename = {}
    for c in df.columns:
        c0 = str(c).strip().lstrip("\ufeff")
        if c0 != c:
            rename[c] = c0
    if rename:
        df = df.rename(columns=rename)
    lower = {str(x).lower().replace(" ", "_"): x for x in df.columns}
    if "timestamp" not in df.columns:
        for key in ("time_stamp", "timestamp", "datetime", "date_time"):
            if key in lower:
                df = df.rename(columns={lower[key]: "timestamp"})
                break
    if "status_id" not in df.columns and "status_type_id" in lower:
        df = df.rename(columns={lower["status_type_id"]: "status_id"})
    return df


def read_care_csv(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        head = fh.readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    df = pd.read_csv(path, sep=sep, low_memory=True, memory_map=True)
    return _downcast_numeric(df)


def load_care_csvs(datasets_dir: str) -> pd.DataFrame:
    csv_files = sorted([f for f in os.listdir(datasets_dir) if f.endswith(".csv")])
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em: {datasets_dir}")

    batch_size = max(1, int(os.environ.get("CARE_CONCAT_BATCH", "4")))
    batch_frames: List[pd.DataFrame] = []
    megas: List[pd.DataFrame] = []

    for f in csv_files:
        path = os.path.join(datasets_dir, f)
        df = read_care_csv(path)
        n = len(df)
        src = pd.Series(
            pd.Categorical.from_codes(np.zeros(n, dtype=np.int8), categories=[f]),
            index=df.index,
            name="source_file",
        )
        df = pd.concat([df, src], axis=1, copy=False)
        batch_frames.append(df)
        if len(batch_frames) >= batch_size:
            part = pd.concat(batch_frames, ignore_index=True, copy=False)
            batch_frames.clear()
            megas.append(part)
            del part
            gc.collect()

    if batch_frames:
        megas.append(pd.concat(batch_frames, ignore_index=True, copy=False))
        batch_frames.clear()
        gc.collect()

    df_all = pd.concat(megas, ignore_index=True, copy=False)
    del megas
    gc.collect()

    df_all = normalize_care_wide_schema(df_all)

    if "timestamp" in df_all.columns:
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
    else:
        raise ValueError(
            "Coluna obrigatoria ausente: timestamp (apos normalizar CARE). "
            f"Primeiras colunas: {list(df_all.columns)[:40]}"
        )

    missing = [c for c in REQUIRED_COLS if c not in df_all.columns]
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")

    return df_all.sort_values("timestamp").reset_index(drop=True)


def add_optional_event_metadata(df: pd.DataFrame, event_info_path: str) -> pd.DataFrame:
    if not os.path.exists(event_info_path):
        return df

    with open(event_info_path, "r", encoding="utf-8", errors="replace") as fh:
        head = fh.readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    event_info = pd.read_csv(event_info_path, sep=sep, low_memory=True)
    if "source_file" in event_info.columns:
        return df.merge(event_info, on="source_file", how="left")
    return df


def temporal_train_calibration_split(df_train_normal: pd.DataFrame, calibration_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df_train_normal.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - calibration_frac))
    train_ae = df_sorted.iloc[:split_idx].copy(deep=False)
    calibration = df_sorted.iloc[split_idx:].copy(deep=False)
    return train_ae, calibration


if not os.path.isdir(DATASETS_DIR):
    raise FileNotFoundError(
        f"Pasta de datasets nao encontrada: {DATASETS_DIR}. "
        "Coloque a pasta CARE_To_Compare na raiz do projeto (veja README.md / Zenodo) ou defina CARE_WIND_FARM_C."
    )

df_raw = load_care_csvs(DATASETS_DIR)
df_raw = add_optional_event_metadata(df_raw, EVENT_INFO_PATH)

train_mask = df_raw["train_test"].astype(str).str.lower().eq("train")
test_mask = df_raw["train_test"].astype(str).str.lower().isin(["test", "prediction"])
normal_mask = ~df_raw["status_id"].isin([3, 4])

df_train_normal = df_raw.loc[train_mask & normal_mask].copy(deep=False)
df_test = df_raw.loc[test_mask].copy(deep=False)

df_train_ae, df_cal = temporal_train_calibration_split(df_train_normal, calibration_frac=0.2)
del df_train_normal

print("Raw:", df_raw.shape)
print("Train AE:", df_train_ae.shape)
print("Calibration:", df_cal.shape)
print("Test:", df_test.shape)


# === Cell 4 ===
test_mask = df_raw["train_test"].astype(str).str.lower().isin(["test", "prediction"])
df_test = df_raw.loc[test_mask].copy(deep=False)
print("Raw:", df_raw.shape)
print("Train AE:", df_train_ae.shape)
print("Calibration:", df_cal.shape)
print("Test:", df_test.shape)

# === Cell 5 ===
# Utilitarios de colunas e labels
META_COLS = {
    "timestamp", "status_id", "train_test", "asset_id", "source_file",
    "event_id", "event_label", "is_padding"
}


def infer_feature_columns(df: pd.DataFrame) -> List[str]:
    feat_cols = []
    for c in df.columns:
        if c in META_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feat_cols.append(c)
    if not feat_cols:
        raise ValueError("Nenhuma feature numerica encontrada.")
    return feat_cols


def infer_binary_labels(df: pd.DataFrame) -> np.ndarray:
    # Prioriza labels explicitos de evento; fallback para status_id
    if "event_label" in df.columns:
        return (~df["event_label"].astype(str).str.lower().eq("normal")).astype(int).to_numpy()
    return df["status_id"].isin([3, 4]).astype(int).to_numpy()


FEATURE_COLS = infer_feature_columns(df_raw)
print("N features numericas:", len(FEATURE_COLS))
del df_raw
gc.collect()


# === Cell 6 ===
# Etapa 2a - DataClipper (antes do sklearn.Pipeline)
import os

class DataClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_q: float = 0.001, upper_q: float = 0.999, exclude_cols: Optional[List[str]] = None):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.exclude_cols = exclude_cols or ["asset_id"]
        self.lower_bounds_: Optional[pd.Series] = None
        self.upper_bounds_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y=None):
        Xdf = pd.DataFrame(X).copy(deep=False)
        cols_to_clip = [c for c in Xdf.columns if c not in self.exclude_cols and pd.api.types.is_numeric_dtype(Xdf[c])]
        self.lower_bounds_ = Xdf[cols_to_clip].quantile(self.lower_q)
        self.upper_bounds_ = Xdf[cols_to_clip].quantile(self.upper_q)
        return self

    def transform(self, X: pd.DataFrame):
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("DataClipper nao foi fitado.")
        Xdf = pd.DataFrame(X).copy()
        cols_to_clip = [c for c in Xdf.columns if c in self.lower_bounds_.index]
        
        lo = self.lower_bounds_[cols_to_clip].to_numpy(dtype=np.float32)
        hi = self.upper_bounds_[cols_to_clip].to_numpy(dtype=np.float32)
        n_rows = len(Xdf)
        chunk = max(1024, int(os.environ.get("CARE_CLIP_CHUNK_ROWS", "32768")))
        
        out = Xdf[cols_to_clip].to_numpy(dtype=np.float32)
        for start in range(0, n_rows, chunk):
            end = min(start + chunk, n_rows)
            block = out[start:end]
            np.clip(block, lo, hi, out=block)
            out[start:end] = block
        
        Xdf[cols_to_clip] = out
        return Xdf


# Sem .copy() profundo: evita duplicar ~8GiB+ por split (vista sobre df_*; clip devolve DataFrame novo)
X_train_ae_raw = df_train_ae.loc[:, FEATURE_COLS + ["asset_id"]]
X_cal_raw = df_cal.loc[:, FEATURE_COLS + ["asset_id"]]
X_test_raw = df_test.loc[:, FEATURE_COLS + ["asset_id"]]

clipper = DataClipper(lower_q=0.001, upper_q=0.999)
clipper.fit(X_train_ae_raw)

X_train_ae_clip = clipper.transform(X_train_ae_raw)
X_cal_clip = clipper.transform(X_cal_raw)
X_test_clip = clipper.transform(X_test_raw)

print("DataClipper aplicado com asset_id preservado.")

# === Cell 7 ===
# Etapa 2b - DataPreprocessor (ordem obrigatoria)
class DuplicateValuesToNan(BaseEstimator, TransformerMixin):
    def __init__(self, value_to_replace: float = 0.0, n_max_duplicates: int = 6):
        self.value_to_replace = value_to_replace
        self.n_max_duplicates = n_max_duplicates

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xdf = pd.DataFrame(X).copy(deep=False)
        for c in Xdf.columns:
            if c == "asset_id": continue
            s = Xdf[c]
            if not pd.api.types.is_numeric_dtype(s):
                continue
            is_val = s.eq(self.value_to_replace)
            run_id = is_val.ne(is_val.shift()).cumsum()
            run_len = is_val.groupby(run_id).transform("sum")
            Xdf.loc[is_val & (run_len >= self.n_max_duplicates), c] = np.nan
        return Xdf


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, max_nan_frac_per_col: float = 0.20):
        self.max_nan_frac_per_col = max_nan_frac_per_col
        self.selected_cols_: List[str] = []

    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X)
        nan_frac = Xdf.isna().mean()
        self.selected_cols_ = nan_frac[nan_frac <= self.max_nan_frac_per_col].index.tolist()
        if "asset_id" not in self.selected_cols_ and "asset_id" in Xdf.columns:
            self.selected_cols_.append("asset_id")
        return self

    def transform(self, X):
        Xdf = pd.DataFrame(X)
        return Xdf[[c for c in self.selected_cols_ if c in Xdf.columns]].copy(deep=False)


class CounterDiffTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, counter_cols: Optional[List[str]] = None, reset_strategy: str = "zero", fill_first: str = "nan"):
        self.counter_cols = counter_cols or []
        self.reset_strategy = reset_strategy
        self.fill_first = fill_first

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xdf = pd.DataFrame(X).copy(deep=False)
        valid_cols = [c for c in self.counter_cols if c in Xdf.columns]
        if not valid_cols:
            return Xdf
            
        if "asset_id" in Xdf.columns:
            diff = Xdf.groupby("asset_id")[valid_cols].diff()
        else:
            diff = Xdf[valid_cols].diff()
            
        if self.reset_strategy == "zero":
            diff = diff.mask(diff < 0, 0)
        if self.fill_first == "nan":
            if "asset_id" in Xdf.columns:
                pass
            else:
                diff.iloc[0] = np.nan
        else:
            diff = diff.fillna(0.0)
            
        Xdf[valid_cols] = diff
        return Xdf


class RollingFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, windows: List[int] = [6], core_keys: List[str] = ["temp", "press", "vib", "speed", "power"]):
        self.windows = windows
        self.core_keys = core_keys
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        Xdf = pd.DataFrame(X).copy(deep=False)
        core_cols = [c for c in Xdf.columns if c != "asset_id" and any(k in c.lower() for k in self.core_keys) and pd.api.types.is_numeric_dtype(Xdf[c])]
        
        if not core_cols or not self.windows:
            return Xdf
            
        new_features = {}
        for w in self.windows:
            if "asset_id" in Xdf.columns:
                rolled = Xdf.groupby("asset_id")[core_cols].rolling(window=w, min_periods=1)
                mean_df = rolled.mean().reset_index(level=0, drop=True)
                std_df = rolled.std().reset_index(level=0, drop=True)
                
                mean_df = mean_df.reindex(Xdf.index)
                std_df = std_df.reindex(Xdf.index).fillna(0)
            else:
                mean_df = Xdf[core_cols].rolling(window=w, min_periods=1).mean()
                std_df = Xdf[core_cols].rolling(window=w, min_periods=1).std().fillna(0)
                
            for c in core_cols:
                new_features[f"{c}_roll{w}_mean"] = mean_df[c]
                new_features[f"{c}_roll{w}_std"] = std_df[c]
                
        if new_features:
            new_df = pd.DataFrame(new_features, index=Xdf.index)
            Xdf = pd.concat([Xdf, new_df], axis=1, copy=False)
            
        return Xdf


class LowUniqueValueFilter(BaseEstimator, TransformerMixin):
    def __init__(self, min_unique_value_count: int = 2, max_col_zero_frac: float = 0.99):
        self.min_unique_value_count = min_unique_value_count
        self.max_col_zero_frac = max_col_zero_frac
        self.selected_cols_: List[str] = []

    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X)
        keep = ["asset_id"] if "asset_id" in Xdf.columns else []
        for c in Xdf.columns:
            if c == "asset_id": continue
            unique_count = Xdf[c].nunique(dropna=True)
            zero_frac = Xdf[c].eq(0).mean(skipna=True)
            if unique_count >= self.min_unique_value_count and zero_frac <= self.max_col_zero_frac:
                keep.append(c)
        self.selected_cols_ = list(set(keep))
        return self

    def transform(self, X):
        Xdf = pd.DataFrame(X)
        return Xdf[[c for c in self.selected_cols_ if c in Xdf.columns]].copy(deep=False)


class AngleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, angle_columns: Optional[List[str]] = None):
        self.angle_columns = angle_columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xdf = pd.DataFrame(X).copy(deep=False)
        for c in self.angle_columns:
            if c in Xdf.columns:
                rad = np.deg2rad(Xdf[c].astype(float))
                Xdf[f"{c}_sin"] = np.sin(rad)
                Xdf[f"{c}_cos"] = np.cos(rad)
                Xdf = Xdf.drop(columns=[c])
        return Xdf


class TimeSeriesImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = 'linear', limit: int = 3):
        self.method = method
        self.limit = limit
        self.fallback_imputer = SimpleImputer(strategy="mean")
        self.cols_to_impute_ = []

    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X)
        self.cols_to_impute_ = [c for c in Xdf.columns if c != "asset_id" and pd.api.types.is_numeric_dtype(Xdf[c])]
        self.fallback_imputer.fit(Xdf[self.cols_to_impute_])
        return self

    def transform(self, X):
        Xdf = pd.DataFrame(X).copy(deep=False)
        if "asset_id" in Xdf.columns:
            def _interp(g):
                return g.interpolate(method=self.method, limit=self.limit, limit_direction="both")
            Xdf[self.cols_to_impute_] = Xdf.groupby("asset_id", group_keys=False)[self.cols_to_impute_].apply(_interp)
        else:
            Xdf[self.cols_to_impute_] = Xdf[self.cols_to_impute_].interpolate(method=self.method, limit=self.limit, limit_direction="both")
            
        if Xdf[self.cols_to_impute_].isna().any().any():
            Xdf[self.cols_to_impute_] = self.fallback_imputer.transform(Xdf[self.cols_to_impute_])
            
        return Xdf


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop: List[str]):
        self.cols_to_drop = cols_to_drop
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        Xdf = pd.DataFrame(X).copy(deep=False)
        to_drop = [c for c in self.cols_to_drop if c in Xdf.columns]
        if to_drop:
            return Xdf.drop(columns=to_drop)
        return Xdf


def infer_counter_cols(cols: List[str]) -> List[str]:
    keys = ["counter", "count", "energy", "cum"]
    return [c for c in cols if any(k in c.lower() for k in keys)]


def infer_angle_cols(cols: List[str]) -> List[str]:
    keys = ["angle", "direction", "yaw", "pitch", "nacelle"]
    return [c for c in cols if any(k in c.lower() for k in keys)]


counter_cols = infer_counter_cols(FEATURE_COLS)
angle_cols = infer_angle_cols(FEATURE_COLS)

preprocessor_pipeline = Pipeline([
    ("duplicate_to_nan", DuplicateValuesToNan(value_to_replace=0.0, n_max_duplicates=6)),
    ("column_selector", ColumnSelector(max_nan_frac_per_col=0.20)),
    ("counter_diff", CounterDiffTransformer(counter_cols=counter_cols, reset_strategy="zero", fill_first="nan")),
    ("rolling_features", RollingFeaturesTransformer(windows=[6])), 
    ("low_unique_filter", LowUniqueValueFilter(min_unique_value_count=2, max_col_zero_frac=0.99)),
    ("angle_transform", AngleTransformer(angle_columns=angle_cols)),
    ("time_series_imputer", TimeSeriesImputer(method="linear", limit=3)),
    ("drop_asset_id", DropColumnsTransformer(cols_to_drop=["asset_id"])),
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
])

X_train_ae_proc = preprocessor_pipeline.fit_transform(X_train_ae_clip)
X_cal_proc = preprocessor_pipeline.transform(X_cal_clip)
X_test_proc = preprocessor_pipeline.transform(X_test_clip)

print("Shapes pos-processamento:", X_train_ae_proc.shape, X_cal_proc.shape, X_test_proc.shape)
del X_train_ae_clip, X_cal_clip, X_test_clip
gc.collect()

# === Cell 8 ===
# Override opcional para dry run (desligado por padrao).
if os.environ.get("AE_DRY_RUN", "0") == "1":
    N_OPTUNA_TRIALS_AE = 1
    print("AE_DRY_RUN=1 -> N_OPTUNA_TRIALS_AE=1")
else:
    print(f"N_OPTUNA_TRIALS_AE={N_OPTUNA_TRIALS_AE}")

# === Cell 9 ===
# Etapa 3 + 6 - Autoencoder Keras e Optuna (obrigatorio)

def build_multilayer_autoencoder(input_dim: int, n_layers: int, code_size: int, learning_rate: float, decay_rate: float):
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = inputs

    for units in [200, 100][: max(1, min(2, n_layers))]:
        x = layers.Dense(units, kernel_initializer="he_normal", use_bias=True)(x)
        x = layers.PReLU()(x)
    if n_layers >= 3:
        x = layers.Dense(50, kernel_initializer="he_normal", use_bias=True)(x)
        x = layers.PReLU()(x)

    bottleneck = layers.Dense(code_size, kernel_initializer="he_normal", use_bias=True, name="code")(x)
    x = layers.PReLU()(bottleneck)

    if n_layers >= 3:
        x = layers.Dense(50, kernel_initializer="he_normal", use_bias=True)(x)
        x = layers.PReLU()(x)
    for units in [100, 200][: max(1, min(2, n_layers))]:
        x = layers.Dense(units, kernel_initializer="he_normal", use_bias=True)(x)
        x = layers.PReLU()(x)

    outputs = layers.Dense(input_dim, activation="linear", name="recon")(x)
    model = keras.Model(inputs, outputs, name="MultilayerAutoencoder")

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=decay_rate,
        staircase=False,
    )
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model


def build_regression_nn_for_trial(input_dim: int, units: int = 32, learning_rate: float = 1e-3):
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(units, activation="relu")(inp)
    out = layers.Dense(1, activation="linear")(x)
    model = keras.Model(inp, out, name="RegressionNNTrial")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


def temporal_split_internal(X: np.ndarray, val_frac: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    n = len(X)
    split = int(n * (1 - val_frac))
    return X[:split], X[split:]


def rmse_per_sample(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))


class ScoreStandardizer:
    """Padroniza anomaly scores (RMSE) via z-score fitado nos dados de treino."""

    def __init__(self):
        self.mu_ = 0.0
        self.sigma_ = 1.0

    def fit(self, scores: np.ndarray) -> "ScoreStandardizer":
        self.mu_ = float(np.mean(scores))
        self.sigma_ = float(np.std(scores))
        if self.sigma_ < 1e-12:
            self.sigma_ = 1.0
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return (scores - self.mu_) / self.sigma_


def smooth_scores(scores: np.ndarray, window: int = 5) -> np.ndarray:
    """Suaviza anomaly scores via moving average centrado."""
    if window <= 1:
        return scores
    return pd.Series(scores).rolling(window, center=True, min_periods=1).mean().values


SMOOTHING_WINDOW = int(os.environ.get("SMOOTHING_WINDOW", "5"))
CRITICALITY_THRESHOLD = int(os.environ.get("CRITICALITY_THRESHOLD", "6"))


def _release_tf_memory():
    keras.backend.clear_session()
    gc.collect()


def objective_ae(trial: optuna.Trial) -> float:
    model = None
    reg_nn_trial = None
    recon_cal = None
    _release_tf_memory()

    try:
        n_layers = trial.suggest_int("n_layers", 1, 4)
        code_size = trial.suggest_int("code_size", 8, 64)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        decay_rate = trial.suggest_float("decay_rate", 0.90, 0.999)
        gamma = trial.suggest_float("gamma", 0.01, 0.5)

        Xtr, Xval = temporal_split_internal(X_train_ae_proc, val_frac=0.25)
        model = build_multilayer_autoencoder(
            input_dim=X_train_ae_proc.shape[1],
            n_layers=n_layers,
            code_size=code_size,
            learning_rate=learning_rate,
            decay_rate=decay_rate,
        )

        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-4, restore_best_weights=True)]
        model.fit(
            Xtr,
            Xtr,
            validation_data=(Xval, Xval),
            epochs=50,
            batch_size=128,
            shuffle=False,
            verbose=0,
            callbacks=cb,
        )

        y_cal = infer_binary_labels(df_cal)
        recon_cal = model.predict(X_cal_proc, verbose=0)
        rmse_cal_raw = rmse_per_sample(X_cal_proc, recon_cal)

        # Padroniza RMSE via z-score (fitado no subset normal da calibracao)
        recon_tr_trial = model.predict(Xtr, verbose=0)
        rmse_tr_trial = rmse_per_sample(Xtr, recon_tr_trial)
        trial_scaler = ScoreStandardizer().fit(rmse_tr_trial)
        rmse_cal = smooth_scores(trial_scaler.transform(rmse_cal_raw), window=SMOOTHING_WINDOW)
        del recon_tr_trial, rmse_tr_trial

        # Treina RegressionNN no proprio trial para avaliar threshold adaptativo real.
        normal_cal_mask = ~df_cal["status_id"].isin([3, 4]).to_numpy()
        X_cal_norm = X_cal_proc[normal_cal_mask]
        rmse_cal_norm = rmse_cal[normal_cal_mask]

        reg_nn_trial = build_regression_nn_for_trial(input_dim=X_cal_proc.shape[1], units=32, learning_rate=1e-3)
        reg_nn_trial.fit(
            X_cal_norm,
            rmse_cal_norm,
            validation_split=0.2,
            epochs=120,
            batch_size=128,
            shuffle=False,
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)],
        )
        rmse_pred_cal = trial_scaler.transform(reg_nn_trial.predict(X_cal_proc, verbose=0).reshape(-1))
        pred = (rmse_cal > (rmse_pred_cal + gamma)).astype(int)

        if len(np.unique(y_cal)) > 1:
            score = fbeta_score(y_cal, pred, beta=2.0, zero_division=0)
        else:
            normal_mask = (y_cal == 0)
            fpr = pred[normal_mask].mean() if normal_mask.any() else 1.0
            score = max(0.0, 1.0 - fpr)

        return float(score)

    except tf.errors.ResourceExhaustedError:
        raise optuna.TrialPruned("Trial podado por ResourceExhaustedError (OOM).")

    finally:
        if recon_cal is not None:
            del recon_cal
        if reg_nn_trial is not None:
            del reg_nn_trial
        if model is not None:
            del model
        _release_tf_memory()


sampler = TPESampler(seed=SEED)
AE_STUDY_DB = os.path.join(ARTIFACTS_DIR, "optuna_ae_study.db")
AE_STUDY_NAME = "ae_keras_pipeline"
AE_STORAGE = f"sqlite:///{AE_STUDY_DB}"
study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    study_name=AE_STUDY_NAME,
    storage=AE_STORAGE,
    load_if_exists=True,
)
already_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
remaining_trials = max(0, N_OPTUNA_TRIALS_AE - already_complete)
print(f"Optuna checkpoint: {already_complete} trials completos em {AE_STUDY_DB}")
try:
    if remaining_trials > 0:
        study.optimize(objective_ae, n_trials=remaining_trials)
    else:
        print("Meta de trials ja atingida; reutilizando estudo salvo.")
except KeyboardInterrupt:
    print("Optuna interrompido manualmente. Checkpoint salvo; pode retomar executando a mesma celula.")

completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
if not completed_trials:
    raise RuntimeError(
        "Nenhum trial concluido no Optuna. Rode novamente a celula de treino do AE para gerar best_params."
    )

best_params = study.best_trial.params
print("Best params:", best_params)

# Limpa grafos/objetos dos trials antes do treino final.
_release_tf_memory()
final_model = build_multilayer_autoencoder(
    input_dim=X_train_ae_proc.shape[1],
    n_layers=best_params["n_layers"],
    code_size=best_params["code_size"],
    learning_rate=best_params["learning_rate"],
    decay_rate=best_params["decay_rate"],
)
Xtr, Xval = temporal_split_internal(X_train_ae_proc, val_frac=0.25)

AE_MODEL_CKPT = os.path.join(ARTIFACTS_DIR, "autoencoder_best.keras")
history = final_model.fit(
    Xtr,
    Xtr,
    validation_data=(Xval, Xval),
    epochs=200,
    batch_size=128,
    shuffle=False,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, min_delta=1e-4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(AE_MODEL_CKPT, monitor="val_loss", save_best_only=True, mode="min", verbose=1),
    ],
)

recon_train = final_model.predict(X_train_ae_proc, verbose=0)
recon_cal = final_model.predict(X_cal_proc, verbose=0)
recon_test = final_model.predict(X_test_proc, verbose=0)

rmse_train_raw = rmse_per_sample(X_train_ae_proc, recon_train)
rmse_cal_raw = rmse_per_sample(X_cal_proc, recon_cal)
rmse_test_raw = rmse_per_sample(X_test_proc, recon_test)
del recon_train, recon_cal, recon_test
gc.collect()

# FIX 1: Padroniza RMSE via z-score fitado no treino (dados normais)
score_scaler = ScoreStandardizer().fit(rmse_train_raw)
rmse_train = smooth_scores(score_scaler.transform(rmse_train_raw), window=SMOOTHING_WINDOW)
rmse_cal = smooth_scores(score_scaler.transform(rmse_cal_raw), window=SMOOTHING_WINDOW)
rmse_test = smooth_scores(score_scaler.transform(rmse_test_raw), window=SMOOTHING_WINDOW)
print(f"Score standardization: mu={score_scaler.mu_:.4f}, sigma={score_scaler.sigma_:.4f}")
print(f"rmse_train (scaled): mean={rmse_train.mean():.4f}, std={rmse_train.std():.4f}")
print(f"rmse_test  (scaled): mean={rmse_test.mean():.4f}, std={rmse_test.std():.4f}")


# === Cell 10 ===
# Etapa 4 - Threshold fixo + adaptativo (RegressionNN)

# Evita NameError quando a celula e executada fora de ordem.
if "rmse_cal" not in globals():
    if "final_model" in globals() and "X_cal_proc" in globals() and "rmse_per_sample" in globals():
        recon_cal_tmp = final_model.predict(X_cal_proc, verbose=0)
        rmse_cal = rmse_per_sample(X_cal_proc, recon_cal_tmp)
        del recon_cal_tmp
        gc.collect()
        print("rmse_cal recomposto a partir do final_model.")
    else:
        raise RuntimeError(
            "Dependencias ausentes para esta etapa. Execute primeiro a celula de treino do AE (celula 10)."
        )

def best_fixed_threshold_by_fbeta(y_true: np.ndarray, scores: np.ndarray, beta: float = 0.5) -> float:
    thresholds = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best_t, best_s = float(np.median(scores)), -1.0
    for t in thresholds:
        pred = (scores > t).astype(int)
        s = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        if s > best_s:
            best_s = s
            best_t = float(t)
    return best_t


y_cal = infer_binary_labels(df_cal)

fixed_threshold_p95 = float(np.quantile(rmse_cal[df_cal["status_id"].isin([0, 1, 2])], 0.95)) if (df_cal["status_id"].isin([0, 1, 2]).any()) else float(np.quantile(rmse_cal, 0.95))
fixed_threshold_fbeta = best_fixed_threshold_by_fbeta(y_cal, rmse_cal, beta=2.0)


def build_regression_nn(input_dim: int, units: int = 32, learning_rate: float = 1e-3):
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(units, activation="relu")(inp)
    out = layers.Dense(1, activation="linear")(x)
    model = keras.Model(inp, out, name="RegressionNN")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


normal_cal_mask = ~df_cal["status_id"].isin([3, 4])
X_cal_norm = X_cal_proc[normal_cal_mask.to_numpy()]
rmse_cal_norm = rmse_cal[normal_cal_mask.to_numpy()]

reg_nn = build_regression_nn(input_dim=X_cal_proc.shape[1], units=32, learning_rate=1e-3)
reg_nn.fit(
    X_cal_norm,
    rmse_cal_norm,
    validation_split=0.2,
    epochs=300,
    batch_size=128,
    shuffle=False,
    verbose=0,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
)

rmse_pred_cal = reg_nn.predict(X_cal_proc, verbose=0).reshape(-1)

# gamma otimizado via Optuna no conjunto de calibracao

def objective_gamma(trial: optuna.Trial) -> float:
    gamma = trial.suggest_float("gamma", 0.01, 0.5)
    pred = (rmse_cal > (rmse_pred_cal + gamma)).astype(int)
    if len(np.unique(y_cal)) > 1:
        return float(fbeta_score(y_cal, pred, beta=2.0, zero_division=0))
    normal_pred_rate = pred[normal_cal_mask.to_numpy()].mean() if normal_cal_mask.any() else 1.0
    return float(max(0.0, 1.0 - normal_pred_rate))


gamma_study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
gamma_study.optimize(objective_gamma, n_trials=N_OPTUNA_TRIALS_GAMMA)

gamma_best = gamma_study.best_trial.params["gamma"]
print("fixed_threshold_p95:", fixed_threshold_p95)
print("fixed_threshold_fbeta:", fixed_threshold_fbeta)
print("gamma_best:", gamma_best)

# === Cell 11 ===
# Etapa 5b - ARCANA (AE congelado, bias no input space)
@tf.function
def arcana_loss(model: keras.Model, x: tf.Tensor, b: tf.Variable, alpha: float):
    x_bias = x + b
    recon = model(x_bias, training=False)
    recon_term = tf.reduce_mean(tf.square(recon - x_bias), axis=1)
    bias_term = tf.reduce_mean(tf.abs(b), axis=1)
    return (1.0 - alpha) * recon_term + alpha * bias_term


def run_arcana_for_sample(
    model: keras.Model,
    x: np.ndarray,
    alpha: float = 0.8,
    num_iter: int = 400,
    lr: float = 1e-3,
    init_x_bias: str = "recon",
) -> np.ndarray:
    x_tf = tf.convert_to_tensor(x.reshape(1, -1), dtype=tf.float32)

    prev_trainable = model.trainable
    model.trainable = False

    if init_x_bias == "recon":
        recon = model.predict(x.reshape(1, -1), verbose=0)
        b0 = recon - x.reshape(1, -1)
    else:
        b0 = np.zeros((1, x.shape[0]), dtype=np.float32)

    b = tf.Variable(b0.astype(np.float32))
    opt = keras.optimizers.Adam(learning_rate=lr)

    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(arcana_loss(model, x_tf, b, alpha=alpha))
        grads = tape.gradient(loss, [b])
        opt.apply_gradients(zip(grads, [b]))

    importance = np.abs(b.numpy().reshape(-1))
    model.trainable = prev_trainable
    return importance


def arcana_on_anomalies(
    model: keras.Model,
    X: np.ndarray,
    anomaly_mask: np.ndarray,
    feature_names: List[str],
    alpha: float = 0.8,
    num_iter: int = 400,
    top_n: int = 15,
    max_samples: int = 100,
) -> pd.DataFrame:
    rows = []
    idxs = np.where(anomaly_mask)[0]
    if len(idxs) > max_samples:
        print(f"ARCANA: limitando de {len(idxs)} para {max_samples} amostras anômalas.")
        idxs = idxs[:max_samples]

    for idx in idxs:
        imp = run_arcana_for_sample(model, X[idx], alpha=alpha, num_iter=num_iter, init_x_bias="recon")
        top_idx = np.argsort(-imp)[:top_n]
        for rank, fi in enumerate(top_idx, start=1):
            rows.append({
                "sample_idx": int(idx),
                "feature": feature_names[fi],
                "importance": float(imp[fi]),
                "rank": rank,
            })
    return pd.DataFrame(rows)

# === Cell 12 ===
# Etapa 5a/5c - Predicao, criticality e CARE score por dataset (source_file)

def predict_with_thresholds(
    rmse: np.ndarray,
    fixed_threshold: float,
    rmse_pred: np.ndarray,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    pred_fixed = (rmse > fixed_threshold).astype(int)
    pred_adapt = (rmse > (rmse_pred + gamma)).astype(int)
    return pred_fixed, pred_adapt


def compute_criticality(alarm_flags: np.ndarray, status_is_anomalous: np.ndarray, threshold: int = 6) -> Tuple[np.ndarray, bool]:
    crit = np.zeros(len(alarm_flags), dtype=int)
    for i in range(1, len(alarm_flags)):
        if status_is_anomalous[i]:
            crit[i] = crit[i - 1]
        elif alarm_flags[i]:
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


def fbeta_safe(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5) -> float:
    if len(np.unique(y_true)) <= 1:
        return 0.0
    return float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))


def care_score_formula(fbeta_mean: float, acc_mean: float, efbeta: float, ws_mean: float, any_alarm: bool) -> float:
    if not any_alarm:
        return 0.0
    if acc_mean < 0.5:
        return float(acc_mean)
    return float((fbeta_mean + ws_mean + efbeta + 2.0 * acc_mean) / 5.0)


y_test = infer_binary_labels(df_test)
rmse_pred_test = reg_nn.predict(X_test_proc, verbose=0).reshape(-1)
pred_fixed_test, pred_adapt_test = predict_with_thresholds(
    rmse_test,
    fixed_threshold=fixed_threshold_fbeta,
    rmse_pred=rmse_pred_test,
    gamma=gamma_best,
)

# Avaliacao por sub-dataset (CSV/source_file), conforme CARE.
df_eval = df_test.copy(deep=False).reset_index(drop=True)
df_eval["y_true"] = y_test
df_eval["pred_adapt"] = pred_adapt_test
df_eval["rmse"] = rmse_test
df_eval["rmse_pred"] = rmse_pred_test

if "is_padding" not in df_eval.columns:
    df_eval["is_padding"] = False

dataset_rows = []
y_event_true = []
y_event_pred = []

for source_file, g in df_eval.groupby("source_file", sort=False):
    gt = g["y_true"].to_numpy().astype(int)
    gp = g["pred_adapt"].to_numpy().astype(int)
    is_padding = g["is_padding"].astype(bool).to_numpy()
    non_padding = ~is_padding

    has_anomaly = bool((gt == 1).any())

    # F1/2 por dataset (sem padding)
    f12_dataset = fbeta_safe(gt[non_padding], gp[non_padding], beta=0.5)

    # Accuracy por dataset normal (somente timestamps normais sem padding)
    normal_mask = (gt == 0) & non_padding
    acc_dataset = float((gp[normal_mask] == 0).sum() / max(1, normal_mask.sum()))

    # Criticality e alarme no nivel dataset
    status_is_anomalous = g["status_id"].isin([3, 4]).to_numpy()
    crit, alarm = compute_criticality(gp.astype(bool), status_is_anomalous, threshold=CRITICALITY_THRESHOLD)

    # WS por dataset anomalo: usar janela de evento quando metadados existirem.
    event_window_mask = non_padding.copy()
    if "event_start" in g.columns and "event_end" in g.columns and "timestamp" in g.columns:
        ts = pd.to_datetime(g["timestamp"], errors="coerce")
        ev_start_series = pd.to_datetime(g["event_start"], errors="coerce")
        ev_end_series = pd.to_datetime(g["event_end"], errors="coerce")
        ev_start = ev_start_series.dropna().iloc[0] if ev_start_series.notna().any() else pd.NaT
        ev_end = ev_end_series.dropna().iloc[0] if ev_end_series.notna().any() else pd.NaT
        if pd.notna(ev_start) and pd.notna(ev_end) and ev_end >= ev_start:
            event_window_mask = ((ts >= ev_start) & (ts <= ev_end)).to_numpy() & non_padding
        elif has_anomaly:
            event_window_mask = ((gt == 1) & non_padding)
    elif has_anomaly:
        event_window_mask = ((gt == 1) & non_padding)

    ws_dataset = weighted_score_earliness(gp[event_window_mask]) if has_anomaly else np.nan

    # PR/ROC por dataset (quando labels supervisionados disponiveis)
    pr_auc_dataset = np.nan
    roc_auc_dataset = np.nan
    if len(np.unique(gt[non_padding])) > 1:
        try:
            pr_auc_dataset = float(average_precision_score(gt[non_padding], g.loc[non_padding, "rmse"].to_numpy()))
            roc_auc_dataset = float(roc_auc_score(gt[non_padding], g.loc[non_padding, "rmse"].to_numpy()))
        except Exception:
            pass

    dataset_rows.append({
        "source_file": source_file,
        "has_anomaly": int(has_anomaly),
        "alarm": int(alarm),
        "F1_2_dataset": f12_dataset,
        "Acc_dataset": acc_dataset,
        "WS_dataset": ws_dataset,
        "criticality_max": int(crit.max()),
        "PR_AUC_dataset": pr_auc_dataset,
        "ROC_AUC_dataset": roc_auc_dataset,
    })

    y_event_true.append(int(has_anomaly))
    y_event_pred.append(int(alarm))

care_by_dataset = pd.DataFrame(dataset_rows)

# Agregacao CARE conforme definicao por dataset
anom_mask_ds = care_by_dataset["has_anomaly"] == 1
normal_mask_ds = care_by_dataset["has_anomaly"] == 0

f12_mean = float(care_by_dataset.loc[anom_mask_ds, "F1_2_dataset"].mean()) if anom_mask_ds.any() else 0.0
ws_mean = float(care_by_dataset.loc[anom_mask_ds, "WS_dataset"].dropna().mean()) if anom_mask_ds.any() else 0.0
acc_mean = float(care_by_dataset.loc[normal_mask_ds, "Acc_dataset"].mean()) if normal_mask_ds.any() else 0.0

efbeta = fbeta_safe(np.array(y_event_true, dtype=int), np.array(y_event_pred, dtype=int), beta=0.5)
any_alarm = bool(np.any(np.array(y_event_pred) == 1))

care_value = care_score_formula(f12_mean, acc_mean, efbeta, ws_mean, any_alarm=any_alarm)

care_results = pd.DataFrame([
    {
        "F1_2": f12_mean,
        "Acc": acc_mean,
        "EF1_2": efbeta,
        "WS": ws_mean,
        "CARE": care_value,
        "n_datasets": int(len(care_by_dataset)),
        "n_anomalous_datasets": int(anom_mask_ds.sum()),
        "n_normal_datasets": int(normal_mask_ds.sum()),
        "gamma": float(gamma_best),
        "fixed_threshold": float(fixed_threshold_fbeta),
    }
])

print(care_results)
care_by_dataset.head()

# === Cell 13 ===
# Etapa 5d - Visualizacoes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(history.history.get("loss", []), label="train_loss")
plt.plot(history.history.get("val_loss", []), label="val_loss")
plt.title("AE loss")
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
plt.hist(rmse_test[y_test == 0], bins=50, alpha=0.6, label="normal")
plt.hist(rmse_test[y_test == 1], bins=50, alpha=0.6, label="anomalo")
plt.axvline(fixed_threshold_fbeta, color="red", linestyle="--", label="fixed_th")
plt.title("RMSE test")
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(rmse_test[:2000], label="rmse")
plt.plot((rmse_pred_test + gamma_best)[:2000], label="adaptive_th", alpha=0.8)
plt.scatter(np.where(pred_adapt_test[:2000] == 1)[0], rmse_test[:2000][pred_adapt_test[:2000] == 1], s=10, label="alarms")
plt.legend()
plt.title("RMSE e threshold adaptativo (recorte)")
plt.show()

# Bar chart dos 4 sub-scores CARE
care_row = care_results.iloc[0]
subscores = ["F1_2", "Acc", "EF1_2", "WS"]
vals = [float(care_row[s]) for s in subscores]
plt.figure(figsize=(7, 4))
plt.bar(subscores, vals)
plt.ylim(0, 1)
plt.title("CARE sub-scores")
plt.show()

# PR/ROC por dataset (quando disponivel)
avail_auc = care_by_dataset.dropna(subset=["PR_AUC_dataset", "ROC_AUC_dataset"])
if not avail_auc.empty:
    top_auc = avail_auc.head(12)
    x = np.arange(len(top_auc))
    w = 0.4
    plt.figure(figsize=(12, 4))
    plt.bar(x - w/2, top_auc["PR_AUC_dataset"], width=w, label="PR_AUC")
    plt.bar(x + w/2, top_auc["ROC_AUC_dataset"], width=w, label="ROC_AUC")
    plt.xticks(x, top_auc["source_file"], rotation=70, ha="right")
    plt.ylim(0, 1)
    plt.title("PR/ROC AUC por dataset (amostra)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Cell 14 ===
# ARCANA sobre anomalias detectadas (adaptativo)
# Compatibilidade ampla: evita slicing direto de Pipeline em versoes antigas de sklearn.
name_pipeline = Pipeline(preprocessor_pipeline.steps[:-2])
_arc_feat = clipper.transform(df_train_ae.loc[:, FEATURE_COLS].head(5))
feature_frame_for_names = name_pipeline.transform(_arc_feat)
feature_names_final = list(feature_frame_for_names.columns)
if len(feature_names_final) != X_test_proc.shape[1]:
    feature_names_final = [f"f_{i}" for i in range(X_test_proc.shape[1])]

arcana_results = arcana_on_anomalies(
    final_model,
    X_test_proc,
    anomaly_mask=(pred_adapt_test == 1),
    feature_names=feature_names_final,
    alpha=0.8,
    num_iter=1000,
    top_n=15,
    max_samples=100,
)

arcana_results.head()

# Bar chart horizontal ARCANA (top-N de uma amostra)
if not arcana_results.empty:
    sample0 = int(arcana_results["sample_idx"].iloc[0])
    top_arc = arcana_results[arcana_results["sample_idx"] == sample0].sort_values("importance", ascending=True)
    plt.figure(figsize=(8, 5))
    plt.barh(top_arc["feature"], top_arc["importance"])
    plt.title(f"ARCANA top features - sample {sample0}")
    plt.tight_layout()
    plt.show()

# === Cell 15 ===
# Etapa final - Exportacao de artefatos
with open(os.path.join(ARTIFACTS_DIR, "preprocessor_pipeline.pkl"), "wb") as f:
    pickle.dump(preprocessor_pipeline, f)

with open(os.path.join(ARTIFACTS_DIR, "data_clipper.pkl"), "wb") as f:
    pickle.dump(clipper, f)

final_model.save(os.path.join(ARTIFACTS_DIR, "autoencoder.h5"))

threshold_params = {
    "fixed_threshold_p95": float(fixed_threshold_p95),
    "fixed_threshold_fbeta": float(fixed_threshold_fbeta),
    "gamma": float(gamma_best),
    "adaptive_formula": "rmse_pred + gamma",
    "regression_nn": {"layers": 1, "units": 32, "optimizer": "Adam"},
    "score_standardization": {"mu": float(score_scaler.mu_), "sigma": float(score_scaler.sigma_)},
    "smoothing_window": SMOOTHING_WINDOW,
    "criticality_threshold": CRITICALITY_THRESHOLD,
}
with open(os.path.join(ARTIFACTS_DIR, "threshold_params.json"), "w", encoding="utf-8") as f:
    json.dump(threshold_params, f, indent=2)

# care_results.csv passa a representar metricas por dataset (source_file).
care_by_dataset.to_csv(os.path.join(ARTIFACTS_DIR, "care_results.csv"), index=False)
arcana_results.to_csv(os.path.join(ARTIFACTS_DIR, "arcana_results.csv"), index=False)

with open(os.path.join(ARTIFACTS_DIR, "best_params.json"), "w", encoding="utf-8") as f:
    json.dump(study.best_trial.params, f, indent=2)

with open(os.path.join(ARTIFACTS_DIR, "care_summary.json"), "w", encoding="utf-8") as f:
    json.dump(care_results.iloc[0].to_dict(), f, indent=2)

print("Artefatos exportados em:", ARTIFACTS_DIR)
