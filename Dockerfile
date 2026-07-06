# =============================================================================
# EduPredict Dockerfile
# Menjalankan FastAPI (port 8000) dan Streamlit (port 8501) dalam satu container
# =============================================================================

# Base image: Python 3.11-slim lebih ringan dan lebih baru dari 3.10
FROM python:3.11-slim

# Metadata container
LABEL maintainer="Ahnaf Fawwaz <A11.2024.15835@mhs.dinus.ac.id>"
LABEL description="EduPredict Sistem Prediksi Kelulusan Mahasiswa"
LABEL version="2.0.0"

# ─── Keamanan: jalankan sebagai non-root user ─────────────────────────────────
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# ─── Direktori kerja ──────────────────────────────────────────────────────────
WORKDIR /app

# ─── Dependensi sistem (curl untuk healthcheck) ───────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# ─── Python dependencies ──────────────────────────────────────────────────────
# Salin requirements.txt duluan agar layer ini di-cache Docker
# (tidak rebuild ulang jika hanya source code yang berubah)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─── Source code ──────────────────────────────────────────────────────────────
COPY --chown=appuser:appgroup . .

# ─── Ganti ke non-root user ───────────────────────────────────────────────────
USER appuser

# ─── Port yang dibuka ─────────────────────────────────────────────────────────
EXPOSE 8000
EXPOSE 8501

# ─── Health check (Docker akan monitor status container secara berkala) ────────
# interval: cek tiap 30 detik
# timeout: maksimal 10 detik per cek
# start-period: tunggu 15 detik saat container baru start
# retries: 3x gagal = unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ─── Entrypoint ───────────────────────────────────────────────────────────────
# FastAPI: 2 workers untuk handle request concurrent
# Streamlit: headless mode, disable telemetry
CMD ["sh", "-c", "\
    uvicorn api_fastapi:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 2 \
        --log-level info & \
    streamlit run app_streamlit_api.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.headless true \
        --browser.gatherUsageStats false \
"]
