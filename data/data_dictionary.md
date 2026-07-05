# Data Dictionary: dataset_kelulusan.csv

Tabel berikut menjelaskan arti dari setiap atribut atau fitur yang digunakan dalam dataset:

## 1. Demografi & Latar Belakang (Demographics & Background)
| Nama Kolom | Tipe Data | Deskripsi & Nilai Kategori |
| :--------- | :-------- | :------------------------- |
| `Marital Status` | Integer | Status pernikahan (1: single, 2: married, 3: widower, 4: divorced, 5: facto union, 6: legally separated). |
| `Nacionality` | Integer | Kewarganegaraan mahasiswa (1: Portuguese, 41: Brazilian, dll). |
| `Gender` | Integer | Jenis kelamin (1: male, 0: female). |
| `Age at enrollment` | Integer | Usia mahasiswa saat pertama kali mendaftar. |
| `Displaced` | Integer | Status pendatang/merantau (1: yes, 0: no). |
| `International` | Integer | Status mahasiswa internasional (1: yes, 0: no). |

## 2. Akademik Pra-Universitas & Pendaftaran (Application & Previous Academic)
| Nama Kolom | Tipe Data | Deskripsi & Nilai Kategori |
| :--------- | :-------- | :------------------------- |
| `Application mode` | Integer | Jalur pendaftaran masuk (1: 1st phase general, 15: International, 42: Transfer, dll). |
| `Application order` | Integer | Urutan pilihan pendaftaran (0: pilihan pertama, hingga 9: pilihan terakhir). |
| `Course` | Integer | Program studi yang diambil (Contoh: 33: Biofuel Production, 9119: Informatics Engineering, dll). |
| `Daytime/evening attendance` | Integer | Waktu perkuliahan (1: daytime / pagi-siang, 0: evening / kelas malam). |
| `Previous qualification` | Integer | Tingkat pendidikan sebelumnya (1: Secondary education, 2: Bachelor's degree, dll). |
| `Previous qualification (grade)`| Continuous| Nilai rata-rata dari kualifikasi pendidikan sebelumnya (rentang 0 - 200). |
| `Admission grade` | Continuous| Nilai ujian masuk / admisi (rentang 0 - 200). |

## 3. Latar Belakang Keluarga & Sosial-Ekonomi (Family & Socio-Economic)
| Nama Kolom | Tipe Data | Deskripsi & Nilai Kategori |
| :--------- | :-------- | :------------------------- |
| `Mother's qualification` | Integer | Tingkat pendidikan ibu (1: Secondary Education, 3: Degree, 34: Unknown, dll). |
| `Father's qualification` | Integer | Tingkat pendidikan ayah (1: Secondary Education, 3: Degree, 34: Unknown, dll). |
| `Mother's occupation` | Integer | Pekerjaan ibu (0: Student, 1: Executive/Director, 9: Unskilled Workers, dll). |
| `Father's occupation` | Integer | Pekerjaan ayah (0: Student, 1: Executive/Director, 9: Unskilled Workers, dll). |
| `Educational special needs` | Integer | Status kebutuhan pendidikan khusus (1: yes, 0: no). |
| `Debtor` | Integer | Status memiliki hutang pendaftaran (1: yes, 0: no). |
| `Tuition fees up to date` | Integer | Status pembayaran uang kuliah lancar/terbaru (1: yes, 0: no). |
| `Scholarship holder` | Integer | Status penerima beasiswa (1: yes, 0: no). |

## 4. Kinerja Akademik Semester 1 & 2 (Academic Performance)
| Nama Kolom | Tipe Data | Deskripsi & Nilai Kategori |
| :--------- | :-------- | :------------------------- |
| `Curricular units 1st sem (credited)` | Integer | Jumlah SKS/mata kuliah yang diakreditasi di semester 1. |
| `Curricular units 1st sem (enrolled)` | Integer | Jumlah SKS/mata kuliah yang diambil di semester 1. |
| `Curricular units 1st sem (evaluations)`| Integer | Jumlah evaluasi/ujian yang diikuti di semester 1. |
| `Curricular units 1st sem (approved)` | Integer | Jumlah SKS/mata kuliah yang lulus di semester 1. |
| `Curricular units 1st sem (grade)` | Integer | Nilai rata-rata di semester 1 (rentang 0 - 20). |
| `Curricular units 1st sem (without evaluations)` | Integer | Jumlah mata kuliah tanpa evaluasi di semester 1. |
| `Curricular units 2nd sem (credited)` | Integer | Jumlah SKS/mata kuliah yang diakreditasi di semester 2. |
| `Curricular units 2nd sem (enrolled)` | Integer | Jumlah SKS/mata kuliah yang diambil di semester 2. |
| `Curricular units 2nd sem (evaluations)`| Integer | Jumlah evaluasi/ujian yang diikuti di semester 2. |
| `Curricular units 2nd sem (approved)` | Integer | Jumlah SKS/mata kuliah yang lulus di semester 2. |
| `Curricular units 2nd sem (grade)` | Integer | Nilai rata-rata di semester 2 (rentang 0 - 20). |
| `Curricular units 2nd sem (without evaluations)` | Integer | Jumlah mata kuliah tanpa evaluasi di semester 2. |

> **Catatan konversi:** Pada aplikasi EduPredict, input SKS Indonesia (0–24) 
> dikonversi ke skala `enrolled/approved` UCI menggunakan rumus proporsional 
> `round(SKS / 24 × 6)`. IPS (0–4) dikonversi ke skala `grade` UCI (0–20) 
> menggunakan piecewise linear mapping.

## 5. Indikator Makroekonomi & Target (Macroeconomics & Target)
| Nama Kolom | Tipe Data | Deskripsi & Nilai Kategori |
| :--------- | :-------- | :------------------------- |
| `Unemployment rate` | Continuous| Tingkat pengangguran (%). |
| `Inflation rate` | Continuous| Tingkat inflasi (%). |
| `GDP` | Continuous| Produk Domestik Bruto (GDP). |
| **`Target`** | **Categorical** | **Status akhir mahasiswa (Klasifikasi 3 kategori: `Dropout`, `Enrolled`, `Graduate`).** |