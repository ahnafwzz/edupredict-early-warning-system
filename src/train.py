"""
train.py
========
Skrip pelatihan utama. Memanggil fungsi di ml_core.py, menjalankan seluruh pipeline
dari load data hingga ekspor model dan laporan.

Cara menjalankan (dari root project):
    python src/train.py

Output otomatis:
    models/  → knn_optimized.joblib, naive_bayes_optimized.joblib, svm_optimized.joblib,
               knn_baseline.joblib, naive_bayes_baseline.joblib, svm_baseline.joblib,
               best_student_graduation_model.joblib, scaler.joblib,
               feature_columns.joblib, best_model_name.txt
    reports/ → audit_dataset.json, all_experiment_results.csv, classification_reports.json
"""

import os
import sys
import json
import joblib
import pandas as pd

# Pastikan src/ terdeteksi ketika dijalankan dari root maupun dari dalam src/
SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from ml_core import (
    load_data,
    audit_data,
    preprocess,
    split_and_scale,
    evaluate_model,
    get_baseline_models,
    train_baseline,
    build_optimization_setup,
    run_optimization,
    get_best_model_name,
    build_classification_reports,
    compute_mi_ranking,
)

# ─── Path ────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(BASE_DIR, 'data',    'dataset_kelulusan.csv')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')


def _banner(step: str, total: int, label: str):
    print(f"\n[{step}/{total}] {label}")
    print("─" * 50)


def main():
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ─── 1. Load & Audit ─────────────────────────────────────────────────
    _banner(1, 6, "Load & Audit Dataset")
    df    = load_data(DATA_PATH)
    audit = audit_data(df)

    print(f"  Dimensi       : {audit['n_rows']} baris x {audit['n_cols']} kolom")
    print(f"  Missing values: {audit['missing_values']}")
    print(f"  Duplikat      : {audit['duplicates']}")
    print(f"  Distribusi target asli:")
    for cls, cnt in audit['target_distribution_original'].items():
        print(f"    {cls}: {cnt}")

    audit_path = os.path.join(REPORTS_DIR, 'audit_dataset.json')
    with open(audit_path, 'w', encoding='utf-8') as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    print(f"\n  → Tersimpan: reports/audit_dataset.json")

    # ─── 2. Preprocessing ────────────────────────────────────────────────
    _banner(2, 6, "Preprocessing")
    X, y, feature_columns = preprocess(df)

    binary_dist = y.value_counts().to_dict()
    print(f"  Distribusi target biner:")
    print(f"    Lulus Tepat Waktu (1)      : {binary_dist.get(1, 0)}")
    print(f"    Tidak Lulus Tepat Waktu (0): {binary_dist.get(0, 0)}")
    print(f"  Proporsi kelas positif       : {y.mean():.2%}")
    print(f"  Jumlah fitur prediktor       : {X.shape[1]}")

    (X_train, X_test,
     y_train, y_test,
     X_train_scaled, X_test_scaled,
     scaler) = split_and_scale(X, y)

    print(f"\n  Train set : {X_train.shape[0]} baris")
    print(f"  Test set  : {X_test.shape[0]} baris")
    print(f"  Proporsi positif - train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")

    # ─── 3. Baseline ─────────────────────────────────────────────────────
    _banner(3, 6, "Baseline KNN / Naive Bayes / SVM")
    baseline_models_init   = get_baseline_models()
    df_baseline, baseline_preds, baseline_trained = train_baseline(
        baseline_models_init, X_train_scaled, y_train, X_test_scaled, y_test
    )
    print("\n  Hasil Baseline:")
    print(df_baseline.round(4).to_string(index=False))

    # ─── 4. Optimasi ─────────────────────────────────────────────────────
    _banner(4, 6, "Optimasi Hyperparameter (GridSearch / RandomizedSearch + CV + Feature Selection)")

    # MI Ranking (informatif, tidak mengubah alur)
    mi_ranking = compute_mi_ranking(X_train_scaled, y_train, feature_columns)
    print("\n  Top 10 fitur (Mutual Information):")
    for fname, score in mi_ranking.head(10).items():
        print(f"    {fname:<55} {score:.4f}")

    pipelines, param_grids = build_optimization_setup(X_train)
    df_optimized, optimized_preds, best_models, best_params = run_optimization(
        pipelines, param_grids,
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        n_splits=5, verbose=True,
    )

    print("\n  Hasil Optimized:")
    print(df_optimized.round(4).to_string(index=False))

    # Perbandingan ringkas baseline vs optimized
    print("\n  Perbandingan Macro-F1 Baseline vs Optimized:")
    for _, row in df_baseline.iterrows():
        name = row['Model']
        b_f1 = row['Macro-F1']
        o_f1 = df_optimized.loc[df_optimized['Model'] == name, 'Macro-F1'].values
        if len(o_f1):
            delta = o_f1[0] - b_f1
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            print(f"    {name:<14} baseline={b_f1:.4f}  optimized={o_f1[0]:.4f}  {arrow} {abs(delta):.4f}")

    # ─── 5. Laporan ──────────────────────────────────────────────────────
    _banner(5, 6, "Menyimpan Laporan")

    # all_experiment_results.csv
    df_b = df_baseline.copy(); df_b['Tahap'] = 'Baseline'
    df_o = df_optimized[['Model', 'CV_Macro-F1', 'Accuracy', 'Precision',
                          'Recall', 'Macro-F1', 'Balanced Acc']].copy()
    df_o['Tahap'] = 'Optimized'
    df_all = pd.concat([df_b, df_o], ignore_index=True)
    results_path = os.path.join(REPORTS_DIR, 'all_experiment_results.csv')
    df_all.to_csv(results_path, index=False)
    print(f"  → Tersimpan: reports/all_experiment_results.csv")

    # classification_reports.json
    all_preds = {}
    for name, y_pred in baseline_preds.items():
        all_preds[f'{name} (Baseline)'] = y_pred
    for name, y_pred in optimized_preds.items():
        all_preds[f'{name} (Optimized)'] = y_pred

    clf_reports = build_classification_reports(all_preds, y_test)

    # Tambahkan best_params ke report
    clf_reports['_best_params'] = best_params

    clf_path = os.path.join(REPORTS_DIR, 'classification_reports.json')
    with open(clf_path, 'w', encoding='utf-8') as f:
        json.dump(clf_reports, f, indent=2, ensure_ascii=False)
    print(f"  → Tersimpan: reports/classification_reports.json")

    # ─── 6. Simpan Model ─────────────────────────────────────────────────
    _banner(6, 6, "Ekspor Model")

    # Baseline models
    for name, model in baseline_trained.items():
        fname = name.lower().replace(' ', '_') + '_baseline.joblib'
        joblib.dump(model, os.path.join(MODELS_DIR, fname))
        print(f"  → {fname}")

    # Optimized models
    for name, model in best_models.items():
        fname = name.lower().replace(' ', '_') + '_optimized.joblib'
        joblib.dump(model, os.path.join(MODELS_DIR, fname))
        print(f"  → {fname}")

    # Scaler & feature columns
    joblib.dump(scaler,          os.path.join(MODELS_DIR, 'scaler.joblib'))
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, 'feature_columns.joblib'))
    print(f"  → scaler.joblib")
    print(f"  → feature_columns.joblib")

    # Model terbaik
    best_name  = get_best_model_name(df_optimized)
    best_model = best_models[best_name]
    best_f1    = df_optimized.loc[df_optimized['Model'] == best_name, 'Macro-F1'].values[0]

    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_student_graduation_model.joblib'))
    with open(os.path.join(MODELS_DIR, 'best_model_name.txt'), 'w') as f:
        f.write(best_name)

    print(f"\n  → best_student_graduation_model.joblib")
    print(f"  → best_model_name.txt")

    # ─── Ringkasan Akhir ─────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  PELATIHAN SELESAI")
    print("=" * 50)
    print(f"  Model terbaik : {best_name}")
    print(f"  Macro-F1 test : {best_f1:.4f}")
    print(f"  File tersimpan di: models/ dan reports/")
    print("=" * 50)


if __name__ == '__main__':
    main()
