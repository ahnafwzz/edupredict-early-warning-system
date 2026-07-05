"""
predict.py
==========
Memuat model, scaler, dan metadata yang sudah disimpan oleh train.py,
lalu melakukan prediksi pada data baru.

Dipakai oleh:
    - app_streamlit.py / app_gradio.py (dipanggil saat inference)
    - CLI: python src/predict.py  (mode demo dengan dummy data)
"""

import os
import sys

import numpy as np
import pandas as pd
import joblib

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

MODELS_DIR = os.path.join(BASE_DIR, 'models')
LABEL_MAP  = {0: 'Tidak Lulus Tepat Waktu', 1: 'Lulus Tepat Waktu'}


# ─── Load Artifacts ──────────────────────────────────────────────────────────
def load_artifacts(models_dir: str = MODELS_DIR) -> tuple:
    """
    Muat model terbaik, scaler, dan metadata dari folder models/.

    Returns
    -------
    model          : fitted sklearn estimator (Pipeline: SelectKBest + clf)
    scaler         : fitted StandardScaler
    feature_columns: list[str]   — urutan fitur yang harus ada di input
    model_name     : str         — nama model terbaik
    """
    model           = joblib.load(os.path.join(models_dir, 'best_student_graduation_model.joblib'))
    scaler          = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
    feature_columns = joblib.load(os.path.join(models_dir, 'feature_columns.joblib'))

    name_path = os.path.join(models_dir, 'best_model_name.txt')
    with open(name_path, 'r') as f:
        model_name = f.read().strip()

    return model, scaler, feature_columns, model_name


def load_model_by_name(name: str, stage: str = 'optimized', models_dir: str = MODELS_DIR):
    """
    Muat model spesifik berdasarkan nama dan stage (baseline/optimized).

    Parameters
    ----------
    name  : 'KNN' | 'Naive Bayes' | 'SVM'
    stage : 'baseline' | 'optimized'
    """
    fname = name.lower().replace(' ', '_') + f'_{stage}.joblib'
    return joblib.load(os.path.join(models_dir, fname))


# ─── Prediksi Tunggal ────────────────────────────────────────────────────────
def predict(
    input_data,
    model=None,
    scaler=None,
    feature_columns=None,
    models_dir: str = MODELS_DIR,
) -> dict:
    """
    Prediksi status kelulusan dari data input.

    Parameters
    ----------
    input_data : dict atau pd.DataFrame (satu atau beberapa baris)
        Key/kolom harus sesuai dengan feature_columns.
    model, scaler, feature_columns : opsional
        Jika None, artifacts dimuat otomatis dari models_dir.

    Returns
    -------
    dict dengan:
        prediction  : int  — 0 atau 1
        label       : str  — 'Tidak Lulus Tepat Waktu' / 'Lulus Tepat Waktu'
        proba       : dict — {'Tidak Lulus Tepat Waktu': float, 'Lulus Tepat Waktu': float}
                             (hanya jika model mendukung predict_proba, mis. KNN)
    Jika input adalah DataFrame multi-baris, kembalikan list of dict.
    """
    if model is None or scaler is None or feature_columns is None:
        model, scaler, feature_columns, _ = load_artifacts(models_dir)

    # ─── Normalisasi input ────────────────────────────────────────────────
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.Series):
        input_df = pd.DataFrame([input_data.to_dict()])
    else:
        input_df = input_data.copy()

    # Pastikan urutan kolom konsisten
    input_df = input_df[feature_columns]

    # ─── Scaling ──────────────────────────────────────────────────────────
    # Scaler di-fit pada data latih; hanya transform di sini.
    input_scaled = scaler.transform(input_df)

    # ─── Prediksi ─────────────────────────────────────────────────────────
    # Pipeline (SelectKBest + model) menerima input yang sudah di-scale.
    preds = model.predict(input_scaled)

    results = []
    for i, pred in enumerate(preds):
        result = {
            'prediction': int(pred),
            'label':      LABEL_MAP[int(pred)],
        }

        # Probabilitas (tersedia untuk KNN, tidak untuk SVM default)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_scaled[i:i+1])[0]
            result['proba'] = {
                'Tidak Lulus Tepat Waktu': round(float(proba[0]), 4),
                'Lulus Tepat Waktu':       round(float(proba[1]), 4),
            }

        results.append(result)

    # Kembalikan dict tunggal jika input satu baris, list jika banyak baris
    return results[0] if len(results) == 1 else results


# ─── Prediksi Batch ──────────────────────────────────────────────────────────
def predict_batch(df_input: pd.DataFrame, model=None, scaler=None, feature_columns=None) -> pd.DataFrame:
    """
    Prediksi untuk seluruh baris dalam DataFrame.

    Returns
    -------
    pd.DataFrame asli dengan kolom tambahan: 'Prediksi_Kode' dan 'Prediksi_Label'.
    """
    results = predict(df_input, model, scaler, feature_columns)
    if isinstance(results, dict):
        results = [results]

    df_out = df_input.copy()
    df_out['Prediksi_Kode']  = [r['prediction'] for r in results]
    df_out['Prediksi_Label'] = [r['label']      for r in results]
    return df_out


# ─── CLI Demo ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Memuat artifacts dari models/ ...")
    try:
        model, scaler, feature_columns, model_name = load_artifacts()
    except FileNotFoundError:
        print("ERROR: File model belum ada. Jalankan `python src/train.py` terlebih dahulu.")
        sys.exit(1)

    print(f"  Model aktif    : {model_name}")
    print(f"  Jumlah fitur   : {len(feature_columns)}")

    # Buat dummy sample via data_generator
    from data_generator import generate_single_sample, generate_batch
    sample = generate_single_sample(feature_columns, seed=7)

    print("\nContoh input (satu baris dummy):")
    for k, v in list(sample.items())[:8]:
        print(f"  {k}: {v}")
    print("  ...")

    result = predict(sample, model, scaler, feature_columns)
    print(f"\nHasil prediksi : {result['label']} (kode={result['prediction']})")
    if 'proba' in result:
        print(f"Probabilitas   :")
        for cls, prob in result['proba'].items():
            print(f"  {cls}: {prob:.4f}")

    # Contoh prediksi batch
    print("\nContoh prediksi batch (5 baris):")
    df_batch = generate_batch(n_samples=5, feature_columns=feature_columns)
    df_result = predict_batch(df_batch, model, scaler, feature_columns)
    print(df_result[['Prediksi_Kode', 'Prediksi_Label']].to_string())
