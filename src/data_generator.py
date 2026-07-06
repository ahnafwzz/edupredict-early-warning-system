"""
data_generator.py
=================
Menghasilkan data dummy yang realistis untuk keperluan pengujian aplikasi
Streamlit/Gradio tanpa harus menggunakan data nyata mahasiswa.

Range nilai setiap fitur mengacu pada nilai min-max dataset asli
(UCI Predict Students' Dropout and Academic Success).

Cara pakai:
    from data_generator import generate_single_sample, generate_batch
    sample = generate_single_sample(feature_columns)         # → dict
    batch  = generate_batch(n_samples=10, feature_columns=.) # → pd.DataFrame
"""

import os
import sys
import numpy as np
import pandas as pd

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

RANDOM_STATE = 42

# ─── Rentang nilai berdasarkan min-max dataset asli ──────────────────────────
# Format:
#   tuple (min, max, dtype)  → nilai kontinu/integer acak dalam rentang
#   list                     → pilih acak dari opsi yang ada (nilai kategorikal)
#
# Catatan: 'Daytime/evening attendance\t' memiliki tab literal pada nama kolom -
# ini artefak dari CSV asli dan harus dijaga konsisten.

FEATURE_META = {
    'Marital status':                           (1,     6,      'int'),
    'Application mode':                         (1,     57,     'int'),
    'Application order':                        (0,     9,      'int'),
    'Course':                                   [33, 171, 8014, 9003, 9070, 9085, 9119,
                                                 9130, 9147, 9238, 9254, 9500, 9556,
                                                 9670, 9773, 9853, 9991],
    'Daytime/evening attendance\t':             (0,     1,      'int'),
    'Previous qualification':                   (1,     43,     'int'),
    'Previous qualification (grade)':           (95.0,  190.0,  'float'),
    'Nacionality':                              (1,     109,    'int'),
    "Mother's qualification":                   (1,     44,     'int'),
    "Father's qualification":                   (1,     44,     'int'),
    "Mother's occupation":                      (0,     194,    'int'),
    "Father's occupation":                      (0,     195,    'int'),
    'Admission grade':                          (95.0,  190.0,  'float'),
    'Displaced':                                (0,     1,      'int'),
    'Educational special needs':                (0,     1,      'int'),
    'Debtor':                                   (0,     1,      'int'),
    'Tuition fees up to date':                  (0,     1,      'int'),
    'Gender':                                   (0,     1,      'int'),
    'Scholarship holder':                       (0,     1,      'int'),
    'Age at enrollment':                        (17,    70,     'int'),
    'International':                            (0,     1,      'int'),
    'Curricular units 1st sem (credited)':      (0,     20,     'int'),
    'Curricular units 1st sem (enrolled)':      (0,     26,     'int'),
    'Curricular units 1st sem (evaluations)':   (0,     45,     'int'),
    'Curricular units 1st sem (approved)':      (0,     26,     'int'),
    'Curricular units 1st sem (grade)':         (0.0,   18.875, 'float'),
    'Curricular units 1st sem (without evaluations)': (0, 12,   'int'),
    'Curricular units 2nd sem (credited)':      (0,     19,     'int'),
    'Curricular units 2nd sem (enrolled)':      (0,     23,     'int'),
    'Curricular units 2nd sem (evaluations)':   (0,     33,     'int'),
    'Curricular units 2nd sem (approved)':      (0,     20,     'int'),
    'Curricular units 2nd sem (grade)':         (0.0,   18.571, 'float'),
    'Curricular units 2nd sem (without evaluations)': (0, 12,   'int'),
    'Unemployment rate':                        (7.6,   16.2,   'float'),
    'Inflation rate':                           (-0.8,  3.7,    'float'),
    'GDP':                                      (-4.06, 3.51,   'float'),
}


def _sample_feature(col_name: str, rng: np.random.Generator) -> object:
    """Generate satu nilai acak untuk satu fitur."""
    if col_name not in FEATURE_META:
        return 0  # fallback aman untuk kolom tidak dikenal

    meta = FEATURE_META[col_name]

    if isinstance(meta, list):
        return int(rng.choice(meta))

    lo, hi, dtype = meta
    if dtype == 'float':
        return round(float(rng.uniform(lo, hi)), 3)
    else:
        return int(rng.integers(lo, hi + 1))


# ─── Fungsi Utama ─────────────────────────────────────────────────────────────
def generate_single_sample(feature_columns: list, seed: int = None) -> dict:
    """
    Generate satu baris data dummy dalam format dict.
    Output langsung bisa dipakai sebagai input ke predict().

    Parameters
    ----------
    feature_columns : list[str]  - urutan fitur dari feature_columns.joblib
    seed            : int        - untuk reproducibility; None = acak penuh

    Returns
    -------
    dict {kolom: nilai}
    """
    rng = np.random.default_rng(seed if seed is not None else RANDOM_STATE)
    return {col: _sample_feature(col, rng) for col in feature_columns}


def generate_batch(
    n_samples: int = 10,
    feature_columns: list = None,
    models_dir: str = None,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate n baris data dummy dalam format DataFrame.

    Parameters
    ----------
    n_samples       : int
    feature_columns : list[str] - jika None, dimuat dari models/feature_columns.joblib
    models_dir      : str       - path ke folder models/ (dipakai jika feature_columns=None)
    seed            : int       - seed utama; tiap baris mendapat sub-seed unik

    Returns
    -------
    pd.DataFrame dengan kolom sesuai feature_columns
    """
    if feature_columns is None:
        if models_dir is None:
            models_dir = os.path.join(BASE_DIR, 'models')
        import joblib
        feature_columns = joblib.load(os.path.join(models_dir, 'feature_columns.joblib'))

    rng = np.random.default_rng(seed if seed is not None else RANDOM_STATE)
    sub_seeds = rng.integers(0, 99_999, size=n_samples).tolist()

    rows = [generate_single_sample(feature_columns, seed=s) for s in sub_seeds]
    return pd.DataFrame(rows, columns=feature_columns)


def generate_realistic_student(feature_columns: list, profile: str = 'at_risk') -> dict:
    """
    Generate profil mahasiswa dummy yang lebih realistis (bukan acak murni).
    Berguna untuk demo aplikasi agar prediksi lebih mudah diinterpretasikan.

    Parameters
    ----------
    profile : 'at_risk'    → mahasiswa dengan sinyal risiko tidak lulus
              'on_track'   → mahasiswa dengan sinyal positif lulus tepat waktu
              'random'     → sama seperti generate_single_sample
    """
    sample = generate_single_sample(feature_columns, seed=99)

    if profile == 'at_risk':
        overrides = {
            'Curricular units 1st sem (approved)':    0,
            'Curricular units 1st sem (grade)':       0.0,
            'Curricular units 2nd sem (approved)':    0,
            'Curricular units 2nd sem (grade)':       0.0,
            'Tuition fees up to date':                0,
            'Debtor':                                 1,
            'Scholarship holder':                     0,
        }
    elif profile == 'on_track':
        overrides = {
            'Curricular units 1st sem (approved)':    6,
            'Curricular units 1st sem (grade)':       14.5,
            'Curricular units 1st sem (enrolled)':    6,
            'Curricular units 2nd sem (approved)':    6,
            'Curricular units 2nd sem (grade)':       14.0,
            'Curricular units 2nd sem (enrolled)':    6,
            'Tuition fees up to date':                1,
            'Debtor':                                 0,
            'Scholarship holder':                     1,
        }
    else:
        return sample

    # Terapkan override hanya pada kolom yang ada di feature_columns
    for col, val in overrides.items():
        if col in feature_columns:
            sample[col] = val

    return sample


# ─── CLI Demo ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Coba muat feature_columns dari models/ jika sudah ada
    models_dir = os.path.join(BASE_DIR, 'models')
    try:
        import joblib
        feature_columns = joblib.load(os.path.join(models_dir, 'feature_columns.joblib'))
        print(f"Feature columns dimuat dari models/ ({len(feature_columns)} fitur)")
    except FileNotFoundError:
        feature_columns = list(FEATURE_META.keys())
        print(f"Menggunakan FEATURE_META fallback ({len(feature_columns)} fitur)")

    print("\n─── generate_single_sample (seed=42) ───")
    sample = generate_single_sample(feature_columns, seed=42)
    for k, v in sample.items():
        print(f"  {k:<55} : {v}")

    print("\n─── generate_batch (5 baris) ───")
    batch = generate_batch(n_samples=5, feature_columns=feature_columns)
    print(batch.to_string())

    print("\n─── generate_realistic_student (at_risk) ───")
    at_risk = generate_realistic_student(feature_columns, profile='at_risk')
    print(f"  Approved 1st sem : {at_risk.get('Curricular units 1st sem (approved)')}")
    print(f"  Grade 1st sem    : {at_risk.get('Curricular units 1st sem (grade)')}")
    print(f"  Tuition up to date: {at_risk.get('Tuition fees up to date')}")

    print("\n─── generate_realistic_student (on_track) ───")
    on_track = generate_realistic_student(feature_columns, profile='on_track')
    print(f"  Approved 1st sem : {on_track.get('Curricular units 1st sem (approved)')}")
    print(f"  Grade 1st sem    : {on_track.get('Curricular units 1st sem (grade)')}")
    print(f"  Tuition up to date: {on_track.get('Tuition fees up to date')}")
