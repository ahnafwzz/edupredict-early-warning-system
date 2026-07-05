"""
api_fastapi.py
==============
EduPredict Backend API REST Prediction Service
Universitas Dian Nuswantoro | UAS Pembelajaran Mesin 2025/2026

Endpoints:
    GET  /              => Informasi API
    GET  /health        => Health check + status model
    GET  /model/info    => Metadata model aktif
    POST /predict       => Prediksi kelulusan mahasiswa

Jalankan:
    uvicorn api_fastapi:app --reload --port 8000
Dokumentasi otomatis:
    http://localhost:8000/docs
"""

import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, os.path.abspath("src"))

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("edupredict")

# ─────────────────────────────────────────────────────────────────────────────
# STORE GLOBAL Model dimuat sekali saat startup
# ─────────────────────────────────────────────────────────────────────────────
store: dict = {
    "model":           None,
    "scaler":          None,
    "feature_columns": None,
    "model_name":      None,
    "loaded":          False,
    "load_error":      None,
}


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN Startup & Shutdown (cara modern FastAPI, menggantikan @on_event)
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # === STARTUP ===
    logger.info("Memuat artifak model Machine Learning...")
    try:
        store["model"]           = joblib.load("models/best_student_graduation_model.joblib")
        store["scaler"]          = joblib.load("models/scaler.joblib")
        store["feature_columns"] = joblib.load("models/feature_columns.joblib")
        store["model_name"]      = "KNN"
        if os.path.exists("models/best_model_name.txt"):
            with open("models/best_model_name.txt") as f:
                store["model_name"] = f.read().strip()
        store["loaded"]          = True
        logger.info(f"Model '{store['model_name']}' berhasil dimuat {len(store['feature_columns'])} fitur.")
    except Exception as e:
        store["load_error"] = str(e)
        logger.error(f"Gagal memuat model: {e}")

    yield  # aplikasi berjalan

    # === SHUTDOWN ===
    logger.info("EduPredict API dimatikan.")


# ─────────────────────────────────────────────────────────────────────────────
# INISIALISASI FASTAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EduPredict API",
    description=(
        "**REST API** untuk Sistem Prediksi Kelulusan Mahasiswa berbasis Machine Learning.\n\n"
        "Dibangun sebagai bagian dari Project UAS Pembelajaran Mesin "
        "Universitas Dian Nuswantoro 2025/2026.\n\n"
        "Input menggunakan **standar perkuliahan Indonesia** (SKS & IPS). "
        "Konversi ke skala dataset UCI dilakukan otomatis di sisi server."
    ),
    version="2.0.0",
    contact={"name": "Ahnaf Fawwaz", "email": "A11.2024.15835@mhs.dinus.ac.id"},
    lifespan=lifespan,
)

# CORS izinkan Streamlit dan browser lokal memanggil API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # production: ganti dengan domain spesifik
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# MIDDLEWARE Request timing log
# ─────────────────────────────────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} => {response.status_code} ({elapsed:.1f}ms)")
    return response


# ─────────────────────────────────────────────────────────────────────────────
# KONVERSI INDONESIA => UCI  (identik dengan app_streamlit.py)
# ─────────────────────────────────────────────────────────────────────────────
def sks_to_uci(sks: int, max_uci: int = 6) -> int:
    return int(round(max(0, sks) / 24.0 * max_uci))


def ips_to_uci(ips: float) -> float:
    """IPS (0–4) => UCI grade dengan piecewise linear mapping."""
    ips = max(0.0, min(4.0, ips))
    if ips < 2.0:    return 10.0
    elif ips < 2.75: return round(11.0 + ((ips - 2.0)  / 0.75) * 1.0, 3)
    elif ips < 3.50: return round(12.0 + ((ips - 2.75) / 0.75) * 1.5, 3)
    else:            return round(13.5 + ((ips - 3.50) / 0.50) * 1.5, 3)


# ─────────────────────────────────────────────────────────────────────────────
# MAPPING KODE UCI  (identik dengan app_streamlit.py)
# ─────────────────────────────────────────────────────────────────────────────
MAP_PENDIDIKAN = {
    "SMA / SMK / Sederajat": 1, "D3 / Diploma": 40, "D4 / Sarjana Terapan": 29,
    "S1 / Sarjana": 2, "S2 / Magister": 4, "S3 / Doktor": 5,
    "Sedang menempuh pendidikan PT": 6, "SMP / Sederajat": 19,
    "SD / Sederajat atau di bawah": 38,
}
MAP_PENDIDIKAN_ORTU = {
    "Tidak / Tidak diketahui": 36, "SD / Sederajat": 38, "SMP / Sederajat": 19,
    "SMA / SMK / Sederajat": 1, "D3 / Diploma": 40, "D4 / Sarjana Terapan": 29,
    "S1 / Sarjana": 2, "S2 / Magister": 4, "S3 / Doktor": 5,
}
MAP_PEKERJAAN = {
    "Tidak bekerja / Tidak diketahui": 99, "Ibu rumah tangga / Petani": 6,
    "Buruh / Pekerja kasar": 9, "Pekerja jasa & perdagangan": 5,
    "Tenaga administrasi / Tata usaha": 4, "Teknisi / Operator": 3,
    "Guru / Pendidik": 123, "Tenaga kesehatan": 122,
    "Spesialis / Profesional": 2, "Wirausaha / Pengusaha": 1,
    "Mahasiswa / Pelajar": 0, "Lainnya / Lain-lain": 99,
}
BASELINE_FEATURES = {
    "Application mode": 17, "Application order": 1, "Course": 9238,
    "Nacionality": 1, "Displaced": 1, "International": 0,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (without evaluations)": 0,
    "Curricular units 2nd sem (credited)": 0,
    "Curricular units 2nd sem (without evaluations)": 0,
    "Unemployment rate": 11.1, "Inflation rate": 1.4, "GDP": 0.32,
}


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS Input & Output
# ─────────────────────────────────────────────────────────────────────────────
VALID_PENDIDIKAN      = list(MAP_PENDIDIKAN.keys())
VALID_PENDIDIKAN_ORTU = list(MAP_PENDIDIKAN_ORTU.keys())
VALID_PEKERJAAN       = list(MAP_PEKERJAAN.keys())


class StudentRequest(BaseModel):
    """Data input mahasiswa menggunakan standar perkuliahan Indonesia."""

    # Akademik Semester 1
    s1_sks_ambil: int   = Field(..., ge=0, le=24,  example=22,   description="SKS diambil Semester 1")
    s1_ips:       float = Field(..., ge=0.0, le=4.0, example=3.58, description="IPS Semester 1 (0.00–4.00)")
    s1_prev_qual: str   = Field(..., example="SMA / SMK / Sederajat",
                                description=f"Pendidikan sebelum masuk PT. Pilihan: {VALID_PENDIDIKAN}")

    # Akademik Semester 2
    s2_sks_ambil: int   = Field(..., ge=0, le=24,  example=20,   description="SKS diambil Semester 2")
    s2_ips:       float = Field(..., ge=0.0, le=4.0, example=3.01, description="IPS Semester 2 (0.00–4.00)")

    # Profil Pribadi
    usia:         int   = Field(..., ge=15, le=60, example=18,    description="Usia saat masuk kuliah")
    gender:       str   = Field(..., example="Laki-laki",          description="Laki-laki / Perempuan")
    status_nikah: str   = Field(..., example="Lajang",             description="Lajang / Menikah / Janda/Duda / Cerai")
    waktu_kuliah: str   = Field(..., example="Reguler (Pagi/Siang)",description="Reguler (Pagi/Siang) / Karyawan (Malam)")
    kebutuhan:    str   = Field(..., example="Tidak Ada",           description="Kebutuhan pendidikan khusus: Ada / Tidak Ada")

    # Finansial
    ukt_lunas:    str   = Field(..., example="Lunas / Tidak Ada Tunggakan",
                                description="Lunas / Tidak Ada Tunggakan  |  Menunggak / Belum Lunas")
    beasiswa:     str   = Field(..., example="Tidak Menerima",     description="Tidak Menerima / Penerima Beasiswa")
    tunggakan:    str   = Field(..., example="Tidak Ada",           description="Tidak Ada / Ada Tunggakan")

    # Orang Tua
    pend_ibu:     str   = Field(..., example="SMA / SMK / Sederajat",
                                description=f"Pendidikan ibu. Pilihan: {VALID_PENDIDIKAN_ORTU}")
    kerja_ibu:    str   = Field(..., example="Tidak bekerja / Tidak diketahui",
                                description=f"Pekerjaan ibu. Pilihan: {VALID_PEKERJAAN}")
    pend_ayah:    str   = Field(..., example="SMA / SMK / Sederajat",
                                description=f"Pendidikan ayah. Pilihan: {VALID_PENDIDIKAN_ORTU}")
    kerja_ayah:   str   = Field(..., example="Buruh / Pekerja kasar",
                                description=f"Pekerjaan ayah. Pilihan: {VALID_PEKERJAAN}")

    # Validator
    @field_validator("s1_prev_qual")
    @classmethod
    def validate_prev_qual(cls, v):
        if v not in MAP_PENDIDIKAN:
            raise ValueError(f"Nilai tidak valid. Pilihan: {VALID_PENDIDIKAN}")
        return v

    @field_validator("pend_ibu", "pend_ayah")
    @classmethod
    def validate_pend_ortu(cls, v):
        if v not in MAP_PENDIDIKAN_ORTU:
            raise ValueError(f"Nilai tidak valid. Pilihan: {VALID_PENDIDIKAN_ORTU}")
        return v

    @field_validator("kerja_ibu", "kerja_ayah")
    @classmethod
    def validate_pekerjaan(cls, v):
        if v not in MAP_PEKERJAAN:
            raise ValueError(f"Nilai tidak valid. Pilihan: {VALID_PEKERJAAN}")
        return v

    model_config = {"json_schema_extra": {"example": {
        "s1_sks_ambil": 22, "s1_ips": 3.58, "s1_prev_qual": "SMA / SMK / Sederajat",
        "s2_sks_ambil": 20, "s2_ips": 3.01, "usia": 18, "gender": "Laki-laki",
        "status_nikah": "Lajang", "waktu_kuliah": "Reguler (Pagi/Siang)",
        "kebutuhan": "Tidak Ada", "ukt_lunas": "Lunas / Tidak Ada Tunggakan",
        "beasiswa": "Tidak Menerima", "tunggakan": "Tidak Ada",
        "pend_ibu": "SMA / SMK / Sederajat", "kerja_ibu": "Tidak bekerja / Tidak diketahui",
        "pend_ayah": "SMA / SMK / Sederajat", "kerja_ayah": "Buruh / Pekerja kasar",
    }}}


class PredictionMetrics(BaseModel):
    probability_graduate_pct: float = Field(description="Probabilitas lulus tepat waktu (%)")
    probability_risk_pct:     float = Field(description="Probabilitas berisiko (%)")


class PredictionResponse(BaseModel):
    success:        bool             = Field(description="Status keberhasilan request")
    prediction:     int              = Field(description="Label prediksi: 1=Lulus, 0=Berisiko")
    label:          str              = Field(description="Label prediksi dalam Bahasa Indonesia")
    interpretation: str              = Field(description="Interpretasi naratif hasil prediksi")
    metrics:        PredictionMetrics
    model_used:     str              = Field(description="Nama algoritma model yang digunakan")
    note:           str              = Field(description="Catatan integritas sistem")


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_name:    Optional[str]
    n_features:    Optional[int]
    error:         Optional[str]


class ModelInfoResponse(BaseModel):
    model_name:     str
    n_features:     int
    feature_names:  list[str]
    supports_proba: bool
    conversion_note: str


# ─────────────────────────────────────────────────────────────────────────────
# HELPER Susun vektor fitur (identik pipeline Streamlit)
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_vector(payload: StudentRequest) -> pd.DataFrame:
    rasio_1 = 1.0 if payload.s1_ips >= 2.50 else (payload.s1_ips / 2.50)
    rasio_2 = 1.0 if payload.s2_ips >= 2.50 else (payload.s2_ips / 2.50)
    s1_lulus = int(payload.s1_sks_ambil * rasio_1)
    s2_lulus = int(payload.s2_sks_ambil * rasio_2)

    data = {
        "Curricular units 1st sem (enrolled)":    sks_to_uci(payload.s1_sks_ambil),
        "Curricular units 1st sem (approved)":    sks_to_uci(s1_lulus),
        "Curricular units 1st sem (evaluations)": sks_to_uci(payload.s1_sks_ambil),
        "Curricular units 1st sem (grade)":       ips_to_uci(payload.s1_ips),
        "Curricular units 2nd sem (enrolled)":    sks_to_uci(payload.s2_sks_ambil),
        "Curricular units 2nd sem (approved)":    sks_to_uci(s2_lulus),
        "Curricular units 2nd sem (evaluations)": sks_to_uci(payload.s2_sks_ambil),
        "Curricular units 2nd sem (grade)":       ips_to_uci(payload.s2_ips),
        "Admission grade":                        127.3,
        "Previous qualification (grade)":         133.1,
        "Previous qualification":                 MAP_PENDIDIKAN.get(payload.s1_prev_qual, 1),
        "Age at enrollment":                      payload.usia,
        "Gender":                                 1 if payload.gender == "Laki-laki" else 0,
        "Marital status":                         {"Lajang": 1, "Menikah": 2, "Janda/Duda": 3, "Cerai": 4}.get(payload.status_nikah, 1),
        "Daytime/evening attendance\t":           1 if "Reguler" in payload.waktu_kuliah else 0,
        "Educational special needs":              1 if payload.kebutuhan == "Ada" else 0,
        "Tuition fees up to date":                0 if "Menunggak" in payload.ukt_lunas else 1,
        "Scholarship holder":                     1 if "Penerima" in payload.beasiswa else 0,
        "Debtor":                                 1 if "Ada" in payload.tunggakan else 0,
        "Mother's qualification":                 MAP_PENDIDIKAN_ORTU.get(payload.pend_ibu, 36),
        "Father's qualification":                 MAP_PENDIDIKAN_ORTU.get(payload.pend_ayah, 36),
        "Mother's occupation":                    MAP_PEKERJAAN.get(payload.kerja_ibu, 99),
        "Father's occupation":                    MAP_PEKERJAAN.get(payload.kerja_ayah, 99),
    }
    data.update(BASELINE_FEATURES)

    feature_columns = store["feature_columns"]
    return pd.DataFrame([data], columns=feature_columns).fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get(
    "/",
    tags=["Sistem"],
    summary="Informasi API",
)
def root():
    """Menampilkan informasi dasar layanan API EduPredict."""
    return {
        "service":     "EduPredict API",
        "version":     "2.0.0",
        "description": "Sistem Prediksi Kelulusan Mahasiswa berbasis Machine Learning",
        "author":      "Ahnaf Fawwaz A11.2024.15835",
        "docs":        "/docs",
        "health":      "/health",
    }


@app.get(
    "/health",
    tags=["Sistem"],
    response_model=HealthResponse,
    summary="Health Check",
)
def health_check():
    """
    Memeriksa status kesehatan server dan ketersediaan model.
    Endpoint ini dapat digunakan oleh monitoring tools atau Docker HEALTHCHECK.
    """
    if store["loaded"]:
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=store["model_name"],
            n_features=len(store["feature_columns"]),
            error=None,
        )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_name=None,
            n_features=None,
            error=store["load_error"],
        ).model_dump(),
    )


@app.get(
    "/model/info",
    tags=["Model"],
    response_model=ModelInfoResponse,
    summary="Metadata Model Aktif",
)
def model_info():
    """Menampilkan detail model yang sedang aktif: nama, jumlah fitur, dan kapabilitas."""
    if not store["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model tidak tersedia.",
        )
    return ModelInfoResponse(
        model_name=store["model_name"],
        n_features=len(store["feature_columns"]),
        feature_names=store["feature_columns"],
        supports_proba=hasattr(store["model"], "predict_proba"),
        conversion_note=(
            "Input SKS (0–24) dan IPS (0.00–4.00) dikonversi ke skala "
            "dataset UCI Portugal secara otomatis sebelum inference."
        ),
    )


@app.post(
    "/predict",
    tags=["Prediksi"],
    response_model=PredictionResponse,
    summary="Prediksi Kelulusan Mahasiswa",
    status_code=status.HTTP_200_OK,
)
def predict(payload: StudentRequest):
    """
    Endpoint utama inference. Menerima data profil mahasiswa dan mengembalikan
    prediksi kelulusan beserta probabilitas dari model Machine Learning.

    - **Prediction 1** => Lulus Tepat Waktu
    - **Prediction 0** => Berisiko Keterlambatan Studi
    """
    if not store["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model Machine Learning tidak tersedia. Pastikan folder models/ sudah ada.",
        )

    try:
        df_input   = build_feature_vector(payload)
        scaled     = store["scaler"].transform(df_input)
        prediction = int(store["model"].predict(scaled)[0])

        # Ekstraksi probabilitas dengan fallback jika model tidak support predict_proba
        if hasattr(store["model"], "predict_proba"):
            proba         = store["model"].predict_proba(scaled)[0]
            classes       = list(store["model"].classes_)
            idx_positive  = classes.index(1) if 1 in classes else -1
            prob_graduate = float(proba[idx_positive]) * 100
        elif hasattr(store["model"], "decision_function"):
            score         = float(store["model"].decision_function(scaled)[0])
            if hasattr(score, "__len__"): score = score[0]
            prob_graduate = float(1 / (1 + np.exp(-score))) * 100
        else:
            prob_graduate = 90.0 if prediction == 1 else 10.0

        prob_risk = 100.0 - prob_graduate

        label = "Lulus Tepat Waktu" if prediction == 1 else "Berisiko Keterlambatan Studi"
        interpretation = (
            "Model mendeteksi pola akademik yang mendukung kelulusan tepat waktu."
            if prediction == 1 else
            "Model mendeteksi indikasi risiko keterlambatan studi. Disarankan intervensi akademik."
        )

        logger.info(
            f"Prediksi: {label} | prob_graduate={prob_graduate:.1f}% | "
            f"IPS_avg={((payload.s1_ips + payload.s2_ips)/2):.2f}"
        )

        return PredictionResponse(
            success=True,
            prediction=prediction,
            label=label,
            interpretation=interpretation,
            metrics=PredictionMetrics(
                probability_graduate_pct=round(prob_graduate, 2),
                probability_risk_pct=round(prob_risk, 2),
            ),
            model_used=store["model_name"],
            note=(
                "Hasil prediksi murni dari algoritma ML tanpa intervensi manual. "
                "Gunakan sebagai decision support, bukan keputusan final akademik."
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Kesalahan internal saat memproses data: {str(e)}",
        )
