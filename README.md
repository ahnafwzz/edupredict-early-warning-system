# EduPredict

**Student Graduation Early Warning System using Machine Learning**

<p align="center">
<img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white" />
<img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" />
<img src="https://img.shields.io/github/last-commit/ahnafwzz/edupredict-early-warning-system" />
<img src="https://img.shields.io/github/repo-size/ahnafwzz/edupredict-early-warning-system" />
<img src="https://img.shields.io/github/stars/ahnafwzz/edupredict-early-warning-system?style=social" />
</p>

> **UAS Pembelajaran Mesin | Semester Genap 2025/2026**
> Universitas Dian Nuswantoro · Fakultas Ilmu Komputer  
> **Ahnaf Fawwaz Arjanta** · A11.2024.15835  
> Dosen Pengampu: **Junta Zeniarja, M.Kom**

---

## Live Demo

| Service | URL |
|---------|-----|
| Streamlit | https://edupredict-ahnaf.streamlit.app |
| FastAPI (Swagger) | http://localhost:8000/docs |
| API Health | http://localhost:8000/health |

---

## Tentang Project

EduPredict adalah aplikasi machine learning yang memprediksi potensi kelulusan mahasiswa, apakah lulus tepat waktu atau berisiko mengalami keterlambatan studi. Prediksi dilakukan berdasarkan data akademik dan finansial semester awal.

Sistem ini berfungsi sebagai early warning system untuk membantu dosen wali dan program studi mengidentifikasi mahasiswa yang membutuhkan pendampingan lebih awal. Hasil prediksi bersifat sebagai pendukung keputusan dan tidak digunakan sebagai penentu status akademik.

Aplikasi terdiri dari antarmuka web (Streamlit), REST API (FastAPI), dan model machine learning yang dibangun dengan scikit-learn.

---

## Tujuan

- Membangun model machine learning untuk prediksi kelulusan mahasiswa.
- Membandingkan performa K-Nearest Neighbors, Naive Bayes, dan Support Vector Machine.
- Mengoptimalkan model menggunakan *feature selection* dan *hyperparameter tuning*.
- Menyediakan aplikasi web yang mudah digunakan.
- Menyediakan REST API agar model dapat diintegrasikan dengan sistem lain.

---

## Features

- Prediksi kelulusan mahasiswa berbasis Machine Learning biner.
- Early Warning System untuk deteksi dini risiko keterlambatan studi.
- Form input interaktif menggunakan standar akademik Indonesia (SKS & IPS).
- Konversi otomatis data SKS/IPS ke representasi skala dataset UCI Portugal di sisi server.
- Perbandingan performa komprehensif antara tiga algoritma: KNN, Naive Bayes, dan SVM.
- Feature selection berbasis **SelectKBest** dengan metode *Mutual Information*.
- Hyperparameter tuning menggunakan **GridSearchCV** dan **RandomizedSearchCV**.
- REST API menggunakan **FastAPI** lengkap dengan dokumentasi otomatis via Swagger UI.
- Antarmuka web yang bersih dan responsif menggunakan **Streamlit**.
- Kontainerisasi dan deployment menggunakan **Docker**.

---

## Tech Stack

| Layer / Kategori | Teknologi |
|------------------|-----------|
| **Programming Language** | Python 3.11 |
| **Machine Learning** | scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Frontend Framework** | Streamlit |
| **REST API Framework** | FastAPI |
| **API Server** | Uvicorn |
| **Model Serialization** | Joblib |
| **Data Validation** | Pydantic |
| **Deployment Engine** | Docker |

---

## Alur Kerja

```
Dataset
   |
   v
Data Audit
   |
   v
Preprocessing
   |
   v
Feature Selection
   |
   v
Train/Test Split
   |
   v
Pelatihan Model (KNN, Naive Bayes, SVM)
   |
   v
Hyperparameter Tuning
   |
   v
Evaluasi Model
   |
   v
Best Model
   |
   v
Deployment
```

---

## Arsitektur Sistem

EduPredict mendukung dua mode penggunaan.

### 1. Mode Monolitik (Streamlit UI)

Mode ini digunakan sebagai aplikasi web mandiri yang langsung diakses oleh pengguna (dosen/admin) untuk melakukan prediksi interaktif melalui antarmuka grafis. Pipeline pemrosesan dan model dijalankan langsung melalui berkas `.joblib`.

```text
User Input (SKS & IPS) ──> Streamlit UI ──> ML Pipeline (.joblib) ──> Visualisasi & Hasil Prediksi
```

### 2. Mode Microservices (FastAPI)

Backend terpisah yang mengekspos fungsi prediksi model melalui REST API sehingga sangat mudah diintegrasikan dengan Sistem Informasi Akademik (SIAKAD) eksternal, aplikasi mobile, maupun sistem desktop lainnya.

```
External System ──> FastAPI Endpoint (/predict) ──> ML Pipeline (.joblib) ──> JSON Response
```

---

## Dataset

Project ini menggunakan dataset **Predict Students' Dropout and Academic Success** dari UCI Machine Learning Repository.

| Atribut | Detail |
|---------|--------|
| Sumber | UCI Machine Learning Repository |
| Jumlah Data | 4.424 mahasiswa |
| Jumlah Fitur | 36 fitur prediktor (data akademik, sosial, dan finansial) |
| Target Awal | Multiclass (Graduate, Dropout, Enrolled) |
| Target Project | Klasifikasi Biner: Lulus Tepat Waktu (1) vs Berisiko Terlambat (0) |
| Missing Value | 0 (Dataset sepenuhnya bersih) |
| Class Balance | Graduate ~50.1%, Non-Graduate ~49.9% |

Dataset ini dipilih karena memiliki data akademik, sosial, dan finansial yang cukup lengkap untuk membangun model prediksi kelulusan.

---

## Preprocessing

Tahapan preprocessing yang dilakukan sebelum pelatihan model:

- Menghapus atribut yang tidak digunakan
- Mengubah target multiclass menjadi biner
- Encoding data kategorikal
- Standardisasi fitur numerik menggunakan StandardScaler
- Membagi data menjadi train/test dengan stratify untuk menjaga proporsi kelas

---

## Feature Selection

Project ini menggunakan SelectKBest dengan metode Mutual Information untuk memilih fitur yang paling berpengaruh. Jumlah fitur yang diuji: 15, 20, dan 36 (seluruh fitur). Pendekatan ini bertujuan mengurangi fitur yang kurang relevan tanpa menurunkan performa model.

---

## Model Machine Learning

Tiga algoritma klasifikasi digunakan dalam project ini:

**K-Nearest Neighbors (KNN)**. Mengklasifikasikan data berdasarkan tetangga terdekat menggunakan jarak Euclidean.

**Naive Bayes**. Pendekatan probabilistik dengan asumsi independensi antar fitur.

**Support Vector Machine (SVM)**. Mencari hyperplane terbaik yang memisahkan dua kelas secara optimal, dikenal memiliki performa baik pada data berdimensi tinggi.

---

## Optimasi Model

Setelah mendapatkan model baseline, setiap algoritma dioptimalkan dengan pendekatan berikut:

| Teknik | Keterangan |
|--------|------------|
| Feature Selection | SelectKBest (Mutual Information) |
| Hyperparameter Tuning | GridSearchCV dan RandomizedSearchCV |
| Cross Validation | Stratified 5-Fold |
| Scoring | Macro F1 |
| Class Balancing | class_weight='balanced' pada SVM |

Seluruh proses menggunakan random_state = 42 agar hasil eksperimen dapat direproduksi.

---

## Hasil Eksperimen

Ringkasan hasil pengujian model:

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| KNN Baseline | 76.72% | 76.57% |
| KNN Optimized | 80.90% | 80.83% |
| Naive Bayes Baseline | 73.22% | 72.63% |
| Naive Bayes Optimized | 73.67% | 72.69% |
| SVM Baseline | **82.15%** | **82.07%** |
| SVM Optimized | 81.69% | 81.64% |

Support Vector Machine memberikan performa terbaik secara keseluruhan. Pada pengujian ini, konfigurasi baseline SVM menghasilkan nilai sedikit lebih tinggi dibandingkan konfigurasi hasil optimasi.

---

## Model Artifacts

File model hasil pelatihan disimpan dalam folder `models/`:

```
models/
   ├── best_student_graduation_model.joblib
   ├── knn_baseline.joblib
   ├── knn_optimized.joblib
   ├── naive_bayes_baseline.joblib
   ├── naive_bayes_optimized.joblib
   ├── svm_baseline.joblib
   ├── svm_optimized.joblib
   ├── scaler.joblib
   ├── feature_columns.joblib
   └── best_model_name.txt
```

Model ini digunakan kembali oleh Streamlit maupun FastAPI sehingga tidak perlu pelatihan ulang setiap kali aplikasi dijalankan.

---

## Laporan

Seluruh hasil eksperimen disimpan di folder `reports/`:

- `audit_dataset.json`
- `all_experiment_results.csv`
- `classification_reports.json`

---

## Struktur Project

```
uas-ml-kelulusan-A11.2024.15835-Ahnaf_Fawwaz/
│
├── data/
│   ├── dataset_kelulusan.csv                # Dataset UCI (separator: ;)
│   ├── data_dictionary.md                   # Deskripsi tiap fitur
│   └── source_dataset.md                    # Sumber dan lisensi dataset
│
├── notebooks/
│   └── graduation_optimization.ipynb        # Notebook utama (Soal 01–05)
│
├── src/
│   ├── ml_core.py                           # Fungsi reusable: load, preprocess, train, evaluate
│   ├── train.py                             # Pipeline pelatihan + ekspor model & laporan
│   ├── predict.py                           # Inference dari model tersimpan
│   └── data_generator.py                    # Generator data dummy untuk pengujian
│
├── models/
│   ├── best_student_graduation_model.joblib # Model SVM terbaik yang siap untuk production
│   ├── knn_baseline.joblib                  # Model K-Nearest Neighbors tanpa optimasi
│   ├── knn_optimized.joblib                 # Model K-Nearest Neighbors hasil tuning
│   ├── naive_bayes_baseline.joblib          # Model Naive Bayes tanpa optimasi
│   ├── naive_bayes_optimized.joblib         # Model Naive Bayes hasil tuning
│   ├── svm_baseline.joblib                  # Model Support Vector Machine tanpa optimasi
│   ├── svm_optimized.joblib                 # Model Support Vector Machine hasil tuning
│   ├── scaler.joblib                        # Objek StandardScaler untuk normalisasi data input
│   ├── feature_columns.joblib               # Daftar urutan kolom fitur untuk standarisasi prediksi
│   └── best_model_name.txt                  # Teks penyimpan nama algoritma model terbaik
│
├── reports/
│   ├── audit_dataset.json                   # Laporan profil data awal (baris, fitur, class balance)
│   ├── all_experiment_results.csv           # Rekap performa akurasi & macro-f1 seluruh eksperimen
│   └── classification_reports.json          # Laporan evaluasi klasifikasi detail (precision, recall)
│
├── presentation/
│   └── presentasi_uas_ml.pdf                # File salindia presentasi hasil eksperimen proyek
│
├── report/
│   └── laporan_uas_ml_kelulusan.pdf         # Dokumen laporan akademik kelengkapan evaluasi penelitian
│
├── README.md                       # Dokumentasi instruksional utama repositori ini 
├── app_streamlit.py                # Aplikasi UI/UX utama (Mode Monolitik)
├── app_streamlit_api.py            # Aplikasi UI/UX yang terintegrasi API (Mode Microservices)
├── api_fastapi.py                  # REST API backend
├── Dockerfile                      # Container konfigurasi untuk deployment
├── requirements.txt                # Daftar library Python (dependencies) yang dibutuhkan proyek
└── Optimization of Machine Learning Model using Grid and Random Search Algorithms for Predicting Student Dropout.pdf                         # Paper utama referensi penelitian
```

---

## Instalasi

### Prasyarat

- Python 3.11 atau lebih baru
- Git
- pip

### Clone Repository

```bash
git clone https://github.com/ahnafwzz/edupredict-early-warning-system.git
cd edupredict-early-warning-system
```

### Install Dependensi

```bash
pip install -r requirements.txt
```

### Latih Model

Jalankan proses pelatihan untuk menghasilkan model dan laporan:

```bash
python src/train.py
```

Atau jalankan notebook `notebooks/graduation_optimization.ipynb`.

---

## Menjalankan Aplikasi
EduPredict mendukung dua skenario penjalanan lokal sesuai kebutuhan arsitektur:

**1. Mode Monolitik (Streamlit Standalone)**
Menjalankan antarmuka web yang memproses model .joblib secara langsung tanpa memerlukan backend terpisah. Sangat cocok untuk pengujian instan.
```bash
streamlit run app_streamlit.py
```

**Mode Microservices (FastAPI + Streamlit Client)**
Mode ini memisahkan REST API (backend) dengan UI (frontend). Buka dua terminal pada root project:

### Terminal 1 (REST API FastAPI Backend):

```bash
uvicorn api_fastapi:app --reload --port 8000
```

### Terminal 2 (Streamlit UI Frontend):

```bash
streamlit run app_streamlit_api.py
```

---

## Layanan Lokal

| Service | URL |
|---------|-----|
| Streamlit | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| API Health | http://localhost:8000/health |
| Model Info | http://localhost:8000/model/info |

---

## Docker

Build image:
```bash
docker build -t edupredict .
```

Jalankan container:
```bash
docker run -p 8000:8000 -p 8501:8501 edupredict
```

---

## REST API

### POST /predict

Melakukan prediksi terhadap data mahasiswa.

**Contoh request:**

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

**Contoh response:**

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
  "note": "Gunakan hasil prediksi sebagai pendukung pengambilan keputusan."
}
```

---

## Reproducibility

Seluruh eksperimen menggunakan seed yang sama:

```python
RANDOM_STATE = 42
```

Seed diterapkan pada train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, dan seluruh model yang mendukung random_state.

Untuk mereproduksi hasil:
```bash
pip install -r requirements.txt
python src/train.py
```

---

## Dependensi

| Library | Kegunaan |
|---------|----------|
| pandas | Pengolahan data |
| numpy | Komputasi numerik |
| scikit-learn | Machine learning |
| joblib | Menyimpan model |
| streamlit | Antarmuka web |
| fastapi | REST API |
| uvicorn | Server FastAPI |
| plotly | Visualisasi interaktif |
| matplotlib | Visualisasi |
| seaborn | Visualisasi statistik |
| requests | Komunikasi API |
| pydantic | Validasi data |

---

## Catatan Etika

- Dataset menggunakan data mahasiswa yang telah dianonimkan dari institusi pendidikan Portugal.
- Aplikasi ini adalah **alat bantu deteksi dini**, bukan penentu kelulusan.
- Model tidak boleh digunakan sebagai keputusan tunggal dalam kebijakan akademik.
- Identitas individu tidak dapat direkonstruksi dari data yang digunakan.

---

## Referensi

1. UCI Machine Learning Repository. Predict Students' Dropout and Academic Success
2. Al Hakim et al. (2026). Optimization of Machine Learning Model using Grid and Random Search Algorithms for Predicting Student Dropout
3. Setiawan et al. (2026). Imbalanced Multi-class Prediction of Student Drop-out and Graduation
4. Turkmenbayev et al. (2025). Machine Learning in Predicting Student Performance in Engineering Programs
5. Malik et al. (2025). Advancing Educational Data Mining through Feature Selection and Classification Techniques
6. Pelima et al. (2024). Predicting University Student Graduation using Academic Performance and Machine Learning

---

## Author

**Ahnaf Fawwaz**
Universitas Dian Nuswantoro
Program Studi Teknik Informatika
Semester Genap 2025/2026

---

## Ucapan Terima Kasih

Project ini dikembangkan sebagai tugas UAS Mata Kuliah Pembelajaran Mesin di Universitas Dian Nuswantoro. Terima kasih kepada dosen pengampu, pihak universitas, serta UCI Machine Learning Repository yang menyediakan dataset untuk penelitian ini.

---

<div align="center">

**EduPredict** <br>
Student Graduation Early Warning System using Machine Learning <br>
Universitas Dian Nuswantoro, 2025/2026

</div>