"""
ml_core.py
==========
Fungsi-fungsi utama yang digunakan bersama oleh train.py, predict.py, dan aplikasi Streamlit/Gradio.
Seluruh logika preprocessing, training, evaluasi, dan optimasi dipusatkan di sini.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix, classification_report
)

# ─── Konstanta global ───────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

LABEL_MAP = {0: 'Tidak Lulus Tepat Waktu', 1: 'Lulus Tepat Waktu'}
TARGET_NAMES = ['Tidak Lulus Tepat Waktu', 'Lulus Tepat Waktu']


# ─── 1. Load Data ────────────────────────────────────────────────────────────
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset dari file CSV dengan separator semicolon.

    Parameters
    ----------
    file_path : str
        Path ke file dataset_kelulusan.csv

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(file_path, sep=';')
    return df


# ─── 2. Audit Data ───────────────────────────────────────────────────────────
def audit_data(df: pd.DataFrame) -> dict:
    """
    Buat ringkasan audit data awal.

    Returns
    -------
    dict berisi dimensi, missing values, duplikat, dan distribusi target.
    Semua nilai dikonversi ke tipe Python native agar bisa di-serialize ke JSON.
    """
    target_dist = {
        str(k): int(v) for k, v in df['Target'].value_counts().items()
    }
    audit = {
        'n_rows': int(df.shape[0]),
        'n_cols': int(df.shape[1]),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicates': int(df.duplicated().sum()),
        'target_distribution_original': target_dist,
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    return audit


# ─── 3. Preprocessing ────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """
    Bersihkan duplikat, binarisasi target, dan pisahkan fitur dari target.

    Mapping target:
        Graduate  → 1  (Lulus Tepat Waktu)
        Dropout   → 0  (Tidak Lulus Tepat Waktu)
        Enrolled  → 0  (Tidak Lulus Tepat Waktu)

    Returns
    -------
    X : pd.DataFrame - fitur prediktor
    y : pd.Series    - target biner
    feature_columns : list[str]
    """
    df = df.drop_duplicates().copy()
    df['Target_Binary'] = df['Target'].apply(lambda x: 1 if x == 'Graduate' else 0)

    feature_columns = [c for c in df.columns if c not in ['Target', 'Target_Binary']]
    X = df[feature_columns].copy()
    y = df['Target_Binary'].copy()

    return X, y, feature_columns


# ─── 4. Split & Scale ────────────────────────────────────────────────────────
def split_and_scale(X: pd.DataFrame, y: pd.Series):
    """
    Stratified train-test split (80/20) lalu StandardScaler.
    Scaler di-fit hanya pada data train untuk menghindari data leakage.

    Returns
    -------
    X_train, X_test, y_train, y_test : split original (unscaled)
    X_train_scaled, X_test_scaled     : hasil scaling
    scaler                            : fitted StandardScaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


# ─── 5. Evaluasi Model ───────────────────────────────────────────────────────
def evaluate_model(y_true, y_pred) -> dict:
    """
    Hitung semua metrik evaluasi yang diminta soal.

    Returns
    -------
    dict dengan Accuracy, Precision, Recall, Macro-F1, Balanced Acc.
    Semua nilai adalah float Python native.
    """
    return {
        'Accuracy':     float(accuracy_score(y_true, y_pred)),
        'Precision':    float(precision_score(y_true, y_pred, average='macro')),
        'Recall':       float(recall_score(y_true, y_pred, average='macro')),
        'Macro-F1':     float(f1_score(y_true, y_pred, average='macro')),
        'Balanced Acc': float(balanced_accuracy_score(y_true, y_pred)),
    }


# ─── 6. Baseline Models ──────────────────────────────────────────────────────
def get_baseline_models() -> dict:
    """
    Kembalikan dict model dengan parameter default sebagai titik acuan baseline.
    Semua model menggunakan parameter scikit-learn default sebelum optimasi.
    """
    return {
        'KNN':         KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'SVM':         SVC(random_state=RANDOM_STATE),
    }


def train_baseline(models: dict, X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Latih seluruh model baseline pada data yang sama, evaluasi di test set.

    Returns
    -------
    df_baseline : pd.DataFrame - hasil metrik semua model baseline
    preds       : dict         - prediksi tiap model pada test set
    trained     : dict         - model yang sudah difit (untuk disimpan)
    """
    results  = []
    preds    = {}
    trained  = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred      = model.predict(X_test_scaled)
        preds[name]   = y_pred
        trained[name] = model
        metrics       = evaluate_model(y_test, y_pred)
        results.append({'Model': name, **metrics})

    df_baseline = pd.DataFrame(results)
    return df_baseline, preds, trained


# ─── 7. Optimasi ─────────────────────────────────────────────────────────────
def build_optimization_setup(X_train: pd.DataFrame):
    """
    Buat Pipeline (SelectKBest + model) dan param grid/dist untuk ketiga model.

    Strategi optimasi:
      - KNN       : GridSearchCV  (ruang kecil, exhaustive aman)
      - Naive Bayes: GridSearchCV
      - SVM       : RandomizedSearchCV (ruang lebih besar, lebih efisien)

    Returns
    -------
    pipelines  : dict[str, Pipeline]
    param_grids: dict[str, dict]
    """
    N = X_train.shape[1]
    k_options = [15, 20, N]

    # ─── Pipelines ─────────────────────────────────────────────────────────
    pipe_knn = Pipeline([
        ('select', SelectKBest(score_func=mutual_info_classif)),
        ('clf',    KNeighborsClassifier()),
    ])
    pipe_nb = Pipeline([
        ('select', SelectKBest(score_func=mutual_info_classif)),
        ('clf',    GaussianNB()),
    ])
    pipe_svm = Pipeline([
        ('select', SelectKBest(score_func=mutual_info_classif)),
        ('clf',    SVC(random_state=RANDOM_STATE)),
    ])

    # ─── Param grids ───────────────────────────────────────────────────────
    param_grid_knn = {
        'select__k':          k_options,
        'clf__n_neighbors':   [3, 5, 7, 9, 11],
        'clf__weights':       ['uniform', 'distance'],
    }
    param_grid_nb = {
        'select__k':          k_options,
        'clf__var_smoothing': np.logspace(-9, -1, 5).tolist(),
        'clf__priors':        [None, [0.5, 0.5]],
    }
    param_dist_svm = {
        'select__k':          k_options,
        'clf__C':             [0.1, 1, 10],
        'clf__kernel':        ['rbf', 'linear'],
        'clf__gamma':         ['scale', 'auto'],
        'clf__class_weight':  ['balanced'],
    }

    pipelines   = {'KNN': pipe_knn, 'Naive Bayes': pipe_nb, 'SVM': pipe_svm}
    param_grids = {'KNN': param_grid_knn, 'Naive Bayes': param_grid_nb, 'SVM': param_dist_svm}

    return pipelines, param_grids


def run_optimization(
    pipelines: dict,
    param_grids: dict,
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
    n_splits: int = 5,
    verbose: bool = True,
):
    """
    Jalankan hyperparameter search untuk KNN, Naive Bayes, dan SVM.
    Scoring utama: macro-F1. Cross-validation: StratifiedKFold.

    Returns
    -------
    df_optimized : pd.DataFrame - metrik test set + CV score tiap model
    preds        : dict         - prediksi optimized tiap model
    best_models  : dict         - estimator terbaik tiap model
    best_params  : dict         - parameter terbaik tiap model
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    searchers = {
        'KNN': GridSearchCV(
            pipelines['KNN'], param_grids['KNN'],
            scoring='f1_macro', cv=cv, n_jobs=-1
        ),
        'Naive Bayes': GridSearchCV(
            pipelines['Naive Bayes'], param_grids['Naive Bayes'],
            scoring='f1_macro', cv=cv, n_jobs=-1
        ),
        'SVM': RandomizedSearchCV(
            pipelines['SVM'], param_grids['SVM'],
            n_iter=20, scoring='f1_macro', cv=cv,
            n_jobs=-1, random_state=RANDOM_STATE
        ),
    }

    best_models = {}
    best_params = {}
    cv_scores   = {}

    for name, searcher in searchers.items():
        if verbose:
            print(f"  Optimasi {name}...", flush=True)
        searcher.fit(X_train_scaled, y_train)
        best_models[name] = searcher.best_estimator_
        best_params[name] = _serialize_params(searcher.best_params_)
        cv_scores[name]   = float(searcher.best_score_)
        if verbose:
            print(f"    Best params  : {searcher.best_params_}")
            print(f"    CV Macro-F1  : {searcher.best_score_:.4f}")

    results = []
    preds   = {}
    for name, model in best_models.items():
        y_pred       = model.predict(X_test_scaled)
        preds[name]  = y_pred
        metrics      = evaluate_model(y_test, y_pred)
        results.append({'Model': name, 'CV_Macro-F1': cv_scores[name], **metrics})

    df_optimized = pd.DataFrame(results)
    return df_optimized, preds, best_models, best_params


# ─── 8. Pilih Model Terbaik ──────────────────────────────────────────────────
def get_best_model_name(df_optimized: pd.DataFrame) -> str:
    """Pilih nama model dengan Macro-F1 tertinggi di test set."""
    return df_optimized.loc[df_optimized['Macro-F1'].idxmax(), 'Model']


# ─── 9. Classification Report ────────────────────────────────────────────────
def build_classification_reports(preds_dict: dict, y_test) -> dict:
    """
    Buat classification_report (output_dict=True) untuk semua model
    dalam preds_dict. Dapat dipakai untuk baseline maupun optimized.

    Parameters
    ----------
    preds_dict : {label: y_pred array}
    y_test     : ground truth

    Returns
    -------
    dict[str, dict] - siap di-dump ke JSON
    """
    reports = {}
    for name, y_pred in preds_dict.items():
        reports[name] = classification_report(
            y_test, y_pred,
            target_names=TARGET_NAMES,
            output_dict=True,
            zero_division=0,
        )
    return reports


# ─── 10. Feature Importance (MI) ─────────────────────────────────────────────
def compute_mi_ranking(X_train_scaled, y_train, feature_columns: list) -> pd.Series:
    """
    Hitung mutual information score tiap fitur terhadap target.

    Returns
    -------
    pd.Series terurut descending
    """
    mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=RANDOM_STATE)
    return pd.Series(mi_scores, index=feature_columns).sort_values(ascending=False)


# ─── Helper ───────────────────────────────────────────────────────────────────
def _serialize_params(params: dict) -> dict:
    """Konversi nilai numpy ke Python native agar JSON-serializable."""
    result = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            result[k] = int(v)
        elif isinstance(v, (np.floating,)):
            result[k] = float(v)
        elif isinstance(v, np.ndarray):
            result[k] = v.tolist()
        else:
            result[k] = v
    return result
