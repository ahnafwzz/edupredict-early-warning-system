# EduPredict — Sistem Prediksi Kelulusan Mahasiswa

> **UAS Pembelajaran Mesin | Semester Genap 2025/2026**  
> Universitas Dian Nuswantoro — Fakultas Ilmu Komputer  
> Ahnaf Fawwaz · A11.2024.15835 · Dosen: Junta Zeniarja, M.Kom

---

## Deskripsi Singkat

**EduPredict** adalah aplikasi _machine learning_ berbasis web yang memprediksi status kelulusan mahasiswa — **lulus tepat waktu** atau **berisiko keterlambatan studi** — berdasarkan data akademik dan finansial semester awal.

Sistem dirancang sebagai alat bantu deteksi dini (_early warning system_) untuk dosen wali dan program studi dalam mengidentifikasi mahasiswa yang membutuhkan intervensi akademik. Prediksi bersifat _decision support_, **bukan** keputusan final kelulusan.

---

## Arsitektur Sistem

```
Input (Streamlit UI)
        ↓
FastAPI /predict  ←→  ML Pipeline (SelectKBest + KNN/NB/SVM)
        ↓                       ↓
 JSON Response            scaler.joblib
        ↓
 Visualisasi Hasil
```

- **Frontend**: Streamlit (`app_streamlit.py`) — form input berbasis standar Indonesia (SKS & IPS)
- **Backend**: FastAPI (`api_fastapi.py`) — REST API dengan Swagger UI otomatis
- **ML Core**: Pipeline scikit-learn (StandardScaler + SelectKBest + Classifier)
- **Deployment**: Docker (`Dockerfile`) — kedua service dalam satu container

---

## Dataset

| Atribut        | Detail |
|----------------|--------|
| Sumber         | [UCI ML Repository — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) |
| Jumlah Baris   | 4.424 mahasiswa |
| Jumlah Fitur   | 36 fitur prediktor |
| Target Asli    | Graduate · Dropout · Enrolled (3 kelas) |
| Target Proyek  | Biner — Lulus Tepat Waktu (1) vs Tidak (0) |
| Missing Values | 0 |
| Class Balance  | ~50.1% Graduate · ~49.9% Non-Graduate |

---

## Algoritma & Optimasi

| Komponen              | Detail |
|-----------------------|--------|
| Model Wajib           | K-Nearest Neighbors, Naive Bayes, Support Vector Machine |
| Feature Selection     | `SelectKBest` (Mutual Information) — k ∈ {15, 20, 36} |
| Hyperparameter Search | GridSearchCV (KNN, NB) + RandomizedSearchCV (SVM) |
| Cross-Validation      | Stratified 5-fold, scoring = macro-F1 |
| Imbalance Handling    | Stratified split + `class_weight='balanced'` pada SVM |
| Metrik Evaluasi       | Accuracy, Precision, Recall, Macro-F1, Balanced Accuracy |

---

## Struktur Folder

```
uas-ml-kelulusan-A11.2024.15835-Ahnaf_Fawwaz/
│
├── data/
│   ├── dataset_kelulusan.csv       # Dataset UCI (separator: ;)
│   ├── data_dictionary.md          # Deskripsi tiap fitur
│   └── source_dataset.md           # Sumber dan lisensi dataset
│
├── notebooks/
│   └── graduation_optimization.ipynb  # Notebook utama (Soal 01–05)
│
├── src/
│   ├── ml_core.py                  # Fungsi reusable: load, preprocess, train, evaluate
│   ├── train.py                    # Pipeline pelatihan + ekspor model & laporan
│   ├── predict.py                  # Inference dari model tersimpan
│   └── data_generator.py           # Generator data dummy untuk pengujian
│
├── models/
│   ├── best_student_graduation_model.joblib
│   ├── knn_baseline.joblib / knn_optimized.joblib
│   ├── naive_bayes_baseline.joblib / naive_bayes_optimized.joblib
│   ├── svm_baseline.joblib / svm_optimized.joblib
│   ├── scaler.joblib
│   ├── feature_columns.joblib
│   └── best_model_name.txt
│
├── reports/
│   ├── audit_dataset.json
│   ├── all_experiment_results.csv
│   └── classification_reports.json
│
├── presentation/
│   └── presentasi_uas_ml.pdf
│
├── report/
│   └── laporan_uas_ml_kelulusan.pdf
│
├── app_streamlit.py                # Aplikasi UI/UX utama
├── api_fastapi.py                  # REST API backend
├── Dockerfile                      # Container untuk deployment
├── requirements.txt
└── README.md
```

---

## Instalasi & Menjalankan

### Prasyarat

- Python 3.10+ (direkomendasikan 3.11)
- pip

### 1. Clone / ekstrak project

```bash
cd uas-ml-kelulusan-A11.2024.15835-Ahnaf_Fawwaz
```

### 2. Install dependensi

```bash
pip install -r requirements.txt
```

### 3. Latih model (generate folder `models/` dan `reports/`)

```bash
python src/train.py
```

Atau jalankan seluruh notebook secara berurutan:
```
notebooks/graduation_optimization.ipynb → Run All
```

### 4. Jalankan aplikasi

Buka **dua terminal terpisah**:

```bash
# Terminal 1 — FastAPI backend
uvicorn api_fastapi:app --reload --port 8000

# Terminal 2 — Streamlit frontend
streamlit run app_streamlit.py
```

| Layanan        | URL                              |
|----------------|----------------------------------|
| Streamlit UI   | http://localhost:8501            |
| FastAPI Docs   | http://localhost:8000/docs       |
| API Health     | http://localhost:8000/health     |
| Model Info     | http://localhost:8000/model/info |

---

## Menjalankan via Docker

```bash
# Build image
docker build -t edupredict .

# Jalankan container
docker run -p 8000:8000 -p 8501:8501 edupredict
```

---

## Contoh API Request

**POST** `http://localhost:8000/predict`

```json
{
  "s1_sks_ambil": 22,
  "s1_ips": 3.58,
  "s1_prev_qual": "SMA / SMK / Sederajat",
  "s2_sks_ambil": 20,
  "s2_ips": 3.01,
  "usia": 18,
  "gender": "Laki-laki",
  "status_nikah": "Lajang",
  "waktu_kuliah": "Reguler (Pagi/Siang)",
  "kebutuhan": "Tidak Ada",
  "ukt_lunas": "Lunas / Tidak Ada Tunggakan",
  "beasiswa": "Tidak Menerima",
  "tunggakan": "Tidak Ada",
  "pend_ibu": "SMA / SMK / Sederajat",
  "kerja_ibu": "Tidak bekerja / Tidak diketahui",
  "pend_ayah": "SMA / SMK / Sederajat",
  "kerja_ayah": "Buruh / Pekerja kasar"
}
```

**Response:**

```json
{
  "success": true,
  "prediction": 1,
  "label": "Lulus Tepat Waktu",
  "interpretation": "Model mendeteksi pola akademik yang mendukung kelulusan tepat waktu.",
  "metrics": {
    "probability_graduate_pct": 80.7,
    "probability_risk_pct": 19.3
  },
  "model_used": "SVM",
  "note": "Hasil prediksi murni dari algoritma ML. Gunakan sebagai decision support."
}
```

---

## Reproducibility

Seluruh eksperimen dapat direproduksi menggunakan:

```python
RANDOM_STATE = 42
```

Seed diterapkan konsisten pada: `train_test_split`, `StratifiedKFold`, `GridSearchCV`, `RandomizedSearchCV`, dan semua model yang mendukung parameter `random_state`.

Untuk mereproduksi environment:

```bash
pip install -r requirements.txt
python src/train.py
```

---

## Catatan Etika

- Dataset menggunakan data mahasiswa yang telah dianonimkan dari institusi pendidikan Portugal.
- Aplikasi ini adalah **alat bantu deteksi dini**, bukan penentu kelulusan.
- Model tidak boleh digunakan sebagai keputusan tunggal dalam kebijakan akademik.
- Identitas individu tidak dapat direkonstruksi dari data yang digunakan.

---

## Referensi Utama

| Penulis | Tahun | Judul | Jurnal | DOI |
|---------|-------|-------|--------|-----|
| Al Hakim et al. | 2026 | Optimization of Machine Learning Model using Grid and Random Search Algorithms for Predicting Student Dropout | JUTIF | 10.52436/1.jutif.2026.7.3.5627 |
| Setiawan et al. | 2026 | Imbalanced Multi-class Prediction of Student Drop-out and Graduation | ERIES Journal | 10.7160/eriesj.2026.190106 |
| Turkmenbayev et al. | 2025 | Machine learning in predicting student performance in engineering programs | Frontiers in Education | 10.3389/feduc.2025.1562586 |
| Malik et al. | 2025 | Advancing EDM through feature selection and classification techniques | Scientific Reports | 10.1038/s41598-025-92324-x |
| Pelima et al. | 2024 | Predicting university student graduation using academic performance and ML | IEEE Access | 10.1109/ACCESS.2024.3361479 |

---

*EduPredict — UAS Pembelajaran Mesin 2025/2026 · Universitas Dian Nuswantoro*
