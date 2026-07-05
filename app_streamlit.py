"""
app_streamlit.py
Aplikasi Prediksi Kelulusan Mahasiswa.
Sistem pendukung keputusan menggunakan Machine Learning.
Input form menggunakan standar sistem perkuliahan Indonesia (SKS dan IPS).
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import requests

sys.path.insert(0, os.path.abspath("src"))

# URL FastAPI backend 
API_BASE_URL = os.getenv("EDUPREDICT_API_URL", "http://localhost:8000")
API_TIMEOUT  = 10 


# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="EduPredict - Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS LIGHT MODE
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"] {
        background-color: #f8f9fb !important;
        color: #111827 !important;
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer { visibility: hidden; }
    [data-testid="stHeader"] { background-color: transparent !important; }

    label, p, span, h1, h2, h3, h4, h5, h6, [data-testid="stWidgetLabel"] p {
        color: #111827 !important;
    }
    div[role="radiogroup"] label span, [data-testid="stMarkdownContainer"] p {
        color: #111827 !important;
    }

    [data-testid="stSidebar"], [data-testid="stSidebarUserContent"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e5e7eb !important;
    }
    [data-testid="stSidebar"] * { color: #111827 !important; }

    input[type="number"], div[data-baseweb="select"] > div, div[data-baseweb="input"] > input {
        background-color: #f0f2f6 !important;
        color: #111827 !important;
        border: 1px solid #e5e7eb !important;
    }
    div[data-baseweb="select"] svg, button svg, [data-testid="stInputNumberButton"] svg {
        fill: #111827 !important;
        color: #111827 !important;
    }

    .streamlit-expanderHeader {
        background-color: #f0f2f6 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        color: #111827 !important;
    }
    [data-testid="stExpander"] summary {
        background-color: #f0f2f6 !important;
        color: #111827 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
    }
    .streamlit-expanderContent {
        background-color: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        color: #111827 !important;
    }

    .dashboard-card {
        background-color: #ffffff; border: 1px solid #e5e7eb;
        border-radius: 12px; padding: 24px; margin-bottom: 16px;
    }
    .metric-box {
        background-color: #f0f2f6; border: 1px solid #e5e7eb;
        border-radius: 10px; padding: 18px 20px; text-align: center;
    }
    .metric-label {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: #4b5563; margin-bottom: 6px;
    }
    .metric-value { font-size: 1.85rem; font-weight: 700; color: #111827; line-height: 1.1; }
    .metric-sub   { font-size: 0.78rem; color: #4b5563; margin-top: 4px; }

    .section-title {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: #4b5563;
        margin-bottom: 12px; padding-bottom: 6px; border-bottom: 1px solid #e5e7eb;
    }

    .result-success {
        background-color: #f0fdf4; border: 1px solid #bbf7d0;
        border-left: 5px solid #22c55e; border-radius: 10px; padding: 20px 24px;
    }
    .result-danger {
        background-color: #fff1f2; border: 1px solid #fecaca;
        border-left: 5px solid #ef4444; border-radius: 10px; padding: 20px 24px;
    }

    .kvbox {
        background-color: #f0f2f6; border: 1px solid #e5e7eb;
        border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;
    }
    .kvlbl { font-size: 0.68rem; color: #4b5563; font-weight: 600; }
    .kvval { font-size: 1rem; font-weight: 700; color: #111827; }

    .note {
        margin-top: 20px; padding: 12px 16px; border-radius: 8px;
        background-color: #f0f2f6; border: 1px solid #e5e7eb;
        font-size: 0.76rem; color: #4b5563; line-height: 1.7;
    }

    .stButton>button {
        background-color: #2563eb !important; color: #ffffff !important;
        border: none !important; border-radius: 8px !important;
        font-weight: 600; font-size: 0.88rem; padding: 9px 18px;
        transition: all 0.15s ease; width: 100%;
    }
    .stButton>button:hover { background-color: #1d4ed8 !important; }
            
    [data-testid="stFormSubmitButton"] > button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        background-color: #1d4ed8 !important;
    }
    
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    ul[role="listbox"] {
        background-color: #ffffff !important;
    }
    li[role="option"] {
        background-color: #ffffff !important;
        color: #111827 !important;
    }
    li[role="option"]:hover {
        background-color: #f0f2f6 !important;
    }
            
    /* Target semua kemungkinan selector sekaligus */
    [data-testid="stInputNumberButton"],
    [data-testid="stInputNumberButton"] > button,
    .stNumberInput button {
        background-color: #e5e7eb !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
    }
    [data-testid="stInputNumberButton"] svg,
    [data-testid="stInputNumberButton"] svg *,
    .stNumberInput button svg * {
        fill: #111827 !important;
        stroke: #111827 !important;
        color: #111827 !important;
    }
            
    hr { border-color: #e5e7eb; margin: 18px 0; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MEMUAT ARTIFAK MODEL DAN LAPORAN
# =============================================================================
@st.cache_resource
def load_ml_models():
    try:
        model_obj      = joblib.load("models/best_student_graduation_model.joblib")
        scaler_obj     = joblib.load("models/scaler.joblib")
        features_list  = joblib.load("models/feature_columns.joblib")
        model_name_str = "KNN"
        if os.path.exists("models/best_model_name.txt"):
            with open("models/best_model_name.txt") as f:
                model_name_str = f.read().strip()
        return model_obj, scaler_obj, features_list, model_name_str, None
    except Exception as error:
        return None, None, None, None, str(error)

@st.cache_data
def load_report_data():
    try:    df_results = pd.read_csv("reports/all_experiment_results.csv")
    except: df_results = pd.DataFrame()
    try:
        with open("reports/classification_reports.json") as f: clf_reports = json.load(f)
    except: clf_reports = {}
    try:
        with open("reports/audit_dataset.json") as f: audit_data = json.load(f)
    except: audit_data = {}
    return df_results, clf_reports, audit_data

def call_predict_api(payload: dict) -> dict:
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Tidak dapat terhubung ke API server. Pastikan FastAPI berjalan di port 8000.")
    except requests.exceptions.Timeout:
        raise RuntimeError("Request ke API timeout setelah 10 detik.")
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e))
        raise RuntimeError(f"API mengembalikan error: {detail}")

def get_api_status() -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return r.json()
    except Exception:
        return None

@st.cache_data(ttl=30)
def _cached_api_status():
    return get_api_status()

model, scaler, feature_columns, model_name, load_error = load_ml_models()
df_reports, classification_reports, dataset_audit = load_report_data()

# =============================================================================
# KONVERSI INDONESIA → UCI
# =============================================================================
def convert_sks_to_uci(sks_input, max_uci=6):
    return int(round(max(0, sks_input) / 24.0 * max_uci))

def convert_ips_to_uci(ips_input):
    """IPS (0–4) → UCI grade dengan pemetaan piecewise linear."""
    ips = max(0.0, min(4.0, ips_input))
    if ips < 2.0:   return 10.0
    elif ips < 2.75: return round(11.0 + ((ips - 2.0)  / 0.75) * 1.0, 3)
    elif ips < 3.50: return round(12.0 + ((ips - 2.75) / 0.75) * 1.5, 3)
    else:            return round(13.5 + ((ips - 3.50) / 0.50) * 1.5, 3)

MAP_PENDIDIKAN = {
    "SMA / SMK / Sederajat":         1,
    "D3 / Diploma":                  40,
    "D4 / Sarjana Terapan":          29,
    "S1 / Sarjana":                   2,
    "S2 / Magister":                  4,
    "S3 / Doktor":                    5,
    "Sedang menempuh pendidikan PT":  6,
    "SMP / Sederajat":               19,
    "SD / Sederajat atau di bawah":  38,
}

MAP_PENDIDIKAN_ORTU = {
    "Tidak / Tidak diketahui":  36,
    "SD / Sederajat":           38,
    "SMP / Sederajat":          19,
    "SMA / SMK / Sederajat":    1,
    "D3 / Diploma":             40,
    "D4 / Sarjana Terapan":     29,
    "S1 / Sarjana":              2,
    "S2 / Magister":             4,
    "S3 / Doktor":               5,
}

MAP_PEKERJAAN = {
    "Tidak bekerja / Tidak diketahui": 99,
    "Ibu rumah tangga / Petani":        6,
    "Buruh / Pekerja kasar":            9,
    "Pekerja jasa & perdagangan":       5,
    "Tenaga administrasi / Tata usaha": 4,
    "Teknisi / Operator":               3,
    "Guru / Pendidik":                123,
    "Tenaga kesehatan":               122,
    "Spesialis / Profesional":          2,
    "Wirausaha / Pengusaha":            1,
    "Mahasiswa / Pelajar":              0,
    "Lainnya / Lain-lain":             99,
}

# Fitur latar (median dataset UCI — tidak ditampilkan ke user)
BASELINE_FEATURES = {
    "Application mode":                               17,
    "Application order":                               1,
    "Course":                                       9238,
    "Nacionality":                                     1,
    "Displaced":                                       1,
    "International":                                   0,
    "Curricular units 1st sem (credited)":             0,
    "Curricular units 1st sem (without evaluations)":  0,
    "Curricular units 2nd sem (credited)":             0,
    "Curricular units 2nd sem (without evaluations)":  0,
    "Unemployment rate":                            11.1,
    "Inflation rate":                                1.4,
    "GDP":                                          0.32,
}

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:24px 0 16px;'>
      <div style='font-size:2.2rem;'>🎓</div>
      <div style='font-size:1.1rem; font-weight:700; color:#111827; margin-top:6px;'>EduPredict</div>
      <div style='font-size:0.76rem; color:#4b5563; margin-top:2px;'>Sistem Deteksi Kelulusan</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    menu_selection = st.radio(
        "Navigasi",
        ["Dashboard", "Analitik Model", "Prediksi"],
        label_visibility="collapsed",
        key="nav_radio",
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    api_status = _cached_api_status()

    if api_status and api_status.get("model_loaded"):
        dot        = "●"
        status_clr = "#16a34a"
        status_bg  = "#f0fdf4"
        status_txt = f"API Online · {api_status.get('model_name', 'Model')}"
    elif api_status:
        dot        = "●"
        status_clr = "#d97706"
        status_bg  = "#fffbeb"
        status_txt = "API Online · Model tidak termuat"
    else:
        dot        = "○"
        status_clr = "#dc2626"
        status_bg  = "#fff1f2"
        status_txt = "API Offline"

    st.markdown(f"""
    <div style='margin-top:12px; font-size:0.73rem; text-align:center; color:{status_clr};
         padding:6px; border-radius:6px; background:{status_bg};
         border:1px solid {status_clr}33;'>
      {dot} {status_txt}
    </div>
    <div style='font-size:0.72rem; color:#4b5563; text-align:center; padding:12px 0 4px;'>
      Ahnaf Fawwaz · A11.2024.15835<br>UAS Pembelajaran Mesin
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# HALAMAN 1 — DASHBOARD
# =============================================================================
if menu_selection == "Dashboard":
    st.markdown("<h2 style='color:#111827; font-weight:700; margin-bottom:4px;'>Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#4b5563; margin-bottom:24px;'>Ringkasan dataset, model aktif, dan informasi sistem.</p>", unsafe_allow_html=True)

    total_rows     = dataset_audit.get("n_rows", 4424)
    total_features = len(feature_columns) if feature_columns else 36
    missing_data   = dataset_audit.get("missing_values", 0)
    target_dist    = dataset_audit.get("target_distribution_original",
                                       {"Graduate": 2209, "Dropout": 1421, "Enrolled": 794})

    col1, col2, col3, col4 = st.columns(4)
    for column, label, value, sub_label in [
        (col1, "Total Data",    f"{total_rows:,}",    "UCI Repository"),
        (col2, "Jumlah Fitur",  str(total_features),  "Fitur Prediktor"),
        (col3, "Missing Values",str(missing_data),     "Dataset Bersih"),
        (col4, "Model Aktif",   model_name or "—",    "Best Macro-F1"),
    ]:
        with column:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{value}</div>
                <div class='metric-sub'>{sub_label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("<div class='section-title'>Distribusi Kelas Target (Dataset Asli)</div>", unsafe_allow_html=True)
        fig_bar = go.Figure(go.Bar(
            x=list(target_dist.keys()), y=list(target_dist.values()),
            marker_color=["#2563eb", "#dc2626", "#9ca3af"],
            text=list(target_dist.values()), textposition="outside",
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#111827", family="Inter", size=12),
            margin=dict(t=30, b=10, l=0, r=0), height=260,
            yaxis=dict(showgrid=True, gridcolor="#d1d5db", tickfont=dict(color="#374151")),
            xaxis=dict(showgrid=False, tickfont=dict(color="#374151")),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-title'>Komposisi Target Biner</div>", unsafe_allow_html=True)
        count_graduate    = target_dist.get("Graduate", 2209)
        count_nongraduate = target_dist.get("Dropout", 1421) + target_dist.get("Enrolled", 794)
        fig_pie = go.Figure(go.Pie(
            labels=["Lulus Tepat Waktu", "Tidak Lulus Tepat Waktu"],
            values=[count_graduate, count_nongraduate],
            hole=0.55, marker_colors=["#2563eb", "#9ca3af"],
            textinfo="percent",
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#111827", family="Inter"),
            margin=dict(t=10, b=10, l=0, r=0), height=260,
            showlegend=True, legend=dict(font=dict(color="#111827")),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown(f"""
    <div class='dashboard-card'>
      <div class='section-title'>Tentang Aplikasi</div>
      <p style='color:#4b5563; line-height:1.75; font-size:0.88rem; margin:0;'>
      Aplikasi ini merupakan implementasi akhir Project UAS mata kuliah
      <b style='color:#111827;'>Pembelajaran Mesin</b> semester genap 2025/2026.
      Model klasifikasi dibangun menggunakan tiga algoritma —
      <b style='color:#111827;'>K-Nearest Neighbors (KNN)</b>,
      <b style='color:#111827;'>Naive Bayes</b>, dan
      <b style='color:#111827;'>Support Vector Machine (SVM)</b> —
      yang dioptimasi melalui GridSearchCV dan RandomizedSearchCV dengan stratified 5-fold cross-validation.<br><br>
      Form prediksi menggunakan <b style='color:#111827;'>standar sistem perkuliahan Indonesia</b>
      (SKS 0–24, IPS 0.00–4.00). Nilai-nilai tersebut dikonversi secara otomatis ke distribusi skala
      dataset UCI sebelum diproses model. Prediksi bersifat <em>decision support</em> untuk
      intervensi akademik dini — bukan keputusan final kelulusan.
      </p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# HALAMAN 2 — ANALITIK MODEL
# =============================================================================
elif menu_selection == "Analitik Model":
    st.markdown("<h2 style='color:#111827; font-weight:700; margin-bottom:4px;'>Analitik Model</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#4b5563; margin-bottom:24px;'>Perbandingan performa baseline dan optimized pada test set.</p>", unsafe_allow_html=True)

    if df_reports.empty:
        st.info("Laporan belum tersedia. Jalankan pelatihan model terlebih dahulu.", icon="ℹ️")
    else:
        list_metrics = ["Accuracy", "Precision", "Recall", "Macro-F1", "Balanced Acc"]
        df_baseline  = df_reports[df_reports["Tahap"] == "Baseline"].copy()
        df_optimized = df_reports[df_reports["Tahap"] == "Optimized"].copy()

        st.markdown("<div class='section-title'>Tabel Perbandingan Baseline vs Optimized</div>", unsafe_allow_html=True)
        display_df = df_reports[["Model", "Tahap"] + [m for m in list_metrics if m in df_reports.columns]].copy()
        for metric in list_metrics:
            if metric in display_df.columns:
                display_df[metric] = (display_df[metric] * 100).round(2).astype(str) + "%"
        st.markdown(
            display_df.to_html(index=False, border=0,
                classes="",
                table_id="tbl_metrik"
            ).replace('<table', '<table style="width:100%;border-collapse:collapse;font-size:0.875rem;color:#111827;"')
            .replace('<th', '<th style="text-align:left;padding:10px 12px;background:#f0f2f6;border-bottom:2px solid #9ca3af;font-weight:600;color:#374151;"')
            .replace('<td', '<td style="padding:10px 12px;border-bottom:1px solid #f3f4f6;color:#111827;"'),
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col_bar, col_radar = st.columns(2)

        with col_bar:
            st.markdown("<div class='section-title'>Macro-F1 — Baseline vs Optimized</div>", unsafe_allow_html=True)
            model_names = df_baseline["Model"].tolist()
            fig_f1 = go.Figure()
            fig_f1.add_trace(go.Bar(
                name="Baseline", x=model_names, y=df_baseline["Macro-F1"].tolist(),
                marker_color="#9ca3af",
                text=[f"{v:.3f}" for v in df_baseline["Macro-F1"]], textposition="outside",
            ))
            if not df_optimized.empty:
                fig_f1.add_trace(go.Bar(
                    name="Optimized", x=model_names, y=df_optimized["Macro-F1"].tolist(),
                    marker_color="#2563eb",
                    text=[f"{v:.3f}" for v in df_optimized["Macro-F1"]], textposition="outside",
                ))
            fig_f1.update_layout(
                barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#111827", family="Inter", size=12),
                margin=dict(t=30, b=10, l=0, r=0), height=300,
                yaxis=dict(showgrid=True, gridcolor="#d1d5db", tickfont=dict(color="#374151")),
                xaxis=dict(showgrid=False, tickfont=dict(color="#374151")),
                legend=dict(font=dict(color="#111827"), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        with col_radar:
            st.markdown("<div class='section-title'>Radar Metrik — Model Optimized</div>", unsafe_allow_html=True)
            radar_metrics = [m for m in ["Accuracy", "Precision", "Recall", "Macro-F1"] if m in df_optimized.columns]
            palette = ["#2563eb", "#6b7280", "#16a34a"]
            fig_radar = go.Figure()
            fill_colors = ["rgba(37,99,235,0.15)", "rgba(107,114,128,0.15)", "rgba(22,163,74,0.15)"]
            for idx, (_, row) in enumerate(df_optimized.iterrows()):
                values = [row[m] * 100 for m in radar_metrics] + [row[radar_metrics[0]] * 100]
                fig_radar.add_trace(go.Scatterpolar(
                    r=values, theta=radar_metrics + [radar_metrics[0]],
                    fill="toself",
                    name=row["Model"],
                    line_color=palette[idx % len(palette)],
                    fillcolor=fill_colors[idx % len(fill_colors)],  # ← ini yang baru
                ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[60, 100], gridcolor="#9ca3af", color="#4b5563",tickfont=dict(color="#4b5563")),
                    angularaxis=dict(color="#111827"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#111827", family="Inter"),
                margin=dict(t=20, b=20, l=60, r=60), height=300,
                legend=dict(font=dict(color="#111827"), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# =============================================================================
# HALAMAN 3 — PREDIKSI
# =============================================================================
elif menu_selection == "Prediksi":
    st.markdown("<h2 style='color:#111827; font-weight:700; margin-bottom:4px;'>Prediksi Kelulusan</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#4b5563; margin-bottom:6px;'>Isi data mahasiswa di bawah ini. SKS Lulus dihitung otomatis dari SKS Diambil dan IPS.</p>", unsafe_allow_html=True)

    if load_error or not model:
        st.error(f"Sistem prediksi belum siap. Jalankan `python src/train.py` terlebih dahulu.\nDetail: {load_error}", icon="❌")
        st.stop()

    with st.form("form_prediksi_utama"):

        # Semester 1
        with st.expander("Data Akademik — Semester 1", expanded=True):
            st.markdown("<p style='font-size:0.82rem; color:#4b5563; margin:0 0 14px;'>Isi berdasarkan KRS dan KHS semester pertama mahasiswa.</p>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                s1_sks_ambil = st.number_input("SKS Diambil", min_value=0, max_value=24, value=22, key="s1e")
            with c2:
                s1_ips = st.number_input("IPS", min_value=0.00, max_value=4.00, value=3.58, step=0.01, key="s1g")
            with c3:
                s1_prev_qual = st.selectbox("Pendidikan Sebelum Kuliah", list(MAP_PENDIDIKAN.keys()), index=0, key="prevq")

        # Semester 2
        with st.expander("Data Akademik — Semester 2", expanded=True):
            st.markdown("<p style='font-size:0.82rem; color:#4b5563; margin:0 0 14px;'>Isi berdasarkan KRS dan KHS semester kedua mahasiswa.</p>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                s2_sks_ambil = st.number_input("SKS Diambil", min_value=0, max_value=24, value=20, key="s2e")
            with c2:
                s2_ips = st.number_input("IPS", min_value=0.00, max_value=4.00, value=3.01, step=0.01, key="s2g")
            with c3:
                st.markdown("<div style='margin-top:35px; font-size:0.8rem; color:#4b5563;'><em>SKS Lulus dan jumlah ujian dihitung dinamis dari SKS dan IPS.</em></div>", unsafe_allow_html=True)

        # Profil Pribadi
        with st.expander("Profil Pribadi", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                usia   = st.number_input("Usia Saat Masuk Kuliah", min_value=16, max_value=60, value=18, key="age")
                gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"], key="gender")
            with c2:
                status_nikah = st.selectbox("Status Pernikahan", ["Lajang", "Menikah", "Janda/Duda", "Cerai"], key="marital")
                waktu_kuliah = st.selectbox("Waktu Kuliah", ["Reguler (Pagi/Siang)", "Karyawan (Malam)"], key="attend")
            with c3:
                kebutuhan = st.selectbox("Kebutuhan Pendidikan Khusus", ["Tidak Ada", "Ada"], key="spneeds")

        # Kondisi Finansial
        with st.expander("Kondisi Finansial"):
            c1, c2, c3 = st.columns(3)
            with c1: ukt_lunas = st.selectbox("Status UKT / SPP", ["Lunas / Tidak Ada Tunggakan", "Menunggak / Belum Lunas"], key="tuition")
            with c2: beasiswa  = st.selectbox("Status Beasiswa", ["Tidak Menerima", "Penerima Beasiswa"], key="scholar")
            with c3: tunggakan = st.selectbox("Riwayat Tunggakan / Utang", ["Tidak Ada", "Ada Tunggakan"], key="debtor")

        # Data Orang Tua
        with st.expander("Data Orang Tua"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div style='font-size:0.78rem; color:#4b5563; font-weight:600; margin-bottom:8px;'>IBU</div>", unsafe_allow_html=True)
                pend_ibu  = st.selectbox("Pendidikan Terakhir Ibu", list(MAP_PENDIDIKAN_ORTU.keys()), index=2, key="mq")
                kerja_ibu = st.selectbox("Pekerjaan Ibu", list(MAP_PEKERJAAN.keys()), index=3, key="mo")
            with c2:
                st.markdown("<div style='font-size:0.78rem; color:#4b5563; font-weight:600; margin-bottom:8px;'>AYAH</div>", unsafe_allow_html=True)
                pend_ayah  = st.selectbox("Pendidikan Terakhir Ayah", list(MAP_PENDIDIKAN_ORTU.keys()), index=2, key="fq")
                kerja_ayah = st.selectbox("Pekerjaan Ayah", list(MAP_PEKERJAAN.keys()), index=2, key="fo")

        st.markdown("<br>", unsafe_allow_html=True)
        btn_submit = st.form_submit_button("Prediksi Kelulusan", use_container_width=True)

    # =========================================================================
    # PROSES INFERENCE
    # =========================================================================
    if btn_submit:
    # Susun payload sesuai schema StudentRequest FastAPI
        api_payload = {
            "s1_sks_ambil": s1_sks_ambil,
            "s1_ips":        s1_ips,
            "s1_prev_qual":  s1_prev_qual,
            "s2_sks_ambil":  s2_sks_ambil,
            "s2_ips":        s2_ips,
            "usia":          usia,
            "gender":        gender,
            "status_nikah":  status_nikah,
            "waktu_kuliah":  waktu_kuliah,
            "kebutuhan":     kebutuhan,
            "ukt_lunas":     ukt_lunas,
            "beasiswa":      beasiswa,
            "tunggakan":     tunggakan,
            "pend_ibu":      pend_ibu,
            "kerja_ibu":     kerja_ibu,
            "pend_ayah":     pend_ayah,
            "kerja_ayah":    kerja_ayah,
        }

        with st.spinner("Mengirim data ke API dan memproses prediksi..."):
            try:
                result     = call_predict_api(api_payload)
                prediction = result["prediction"]        # 0 atau 1
                prob_lulus = result["metrics"]["probability_graduate_pct"]
                prob_risiko= result["metrics"]["probability_risk_pct"]
                model_used = result.get("model_used", model_name or "ML")
                ada_proba  = True
            except RuntimeError as e:
                st.error(f"**Prediksi gagal:** {e}", icon="❌")
                st.info("Jalankan FastAPI terlebih dahulu: `uvicorn api_fastapi:app --port 8000`")
                st.stop()

        # ── Kalkulasi SKS lulus (hanya untuk tampilan ringkasan) ──────────────
        rasio_1     = 1.0 if s1_ips >= 2.50 else (s1_ips / 2.50)
        rasio_2     = 1.0 if s2_ips >= 2.50 else (s2_ips / 2.50)
        s1_sks_lulus = int(s1_sks_ambil * rasio_1)
        s2_sks_lulus = int(s2_sks_ambil * rasio_2)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Hasil Prediksi</div>", unsafe_allow_html=True)

        col_grafik, col_teks = st.columns([1, 1.4])

        with col_grafik:
            warna = "#16a34a" if prob_lulus >= 50 else "#dc2626"
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=prob_lulus,
                number={"suffix": "%", "font": {"size": 36, "color": "#111827", "family": "Inter"}},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#e5e7eb",
                            "tickvals": [0, 20, 40, 60, 80, 100]},
                    "bar":  {"color": warna, "thickness": 0.25},
                    "bgcolor": "#f0f2f6", "borderwidth": 0,
                    "steps": [{"range": [0, 49.9],  "color": "#fecaca"},
                            {"range": [50, 100], "color": "#bbf7d0"}],
                    "threshold": {"line": {"color": warna, "width": 3},
                                "thickness": 0.75, "value": prob_lulus},
                },
            ))
            fig_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#111827", "family": "Inter"},
                height=220, margin=dict(l=30, r=40, t=20, b=10),
            )
            st.plotly_chart(fig_g, use_container_width=True)
            st.markdown(
                f"<div style='text-align:center; font-size:0.75rem; color:#4b5563; margin-top:-10px;'>"
                f"Probabilitas via {model_used} · FastAPI Backend</div>",
                unsafe_allow_html=True,
            )

        with col_teks:
            rata_rata_ips     = (s1_ips + s2_ips) / 2
            total_sks_diambil = s1_sks_ambil + s2_sks_ambil
            total_sks_gagal   = (s1_sks_ambil - s1_sks_lulus) + (s2_sks_ambil - s2_sks_lulus)

            analisis_ips = (
                f"<li>IPS rata-rata {rata_rata_ips:.2f} — kategori {'tinggi, mendukung kestabilan akademik' if rata_rata_ips >= 3.25 else 'wajar, disarankan evaluasi berkala' if rata_rata_ips >= 2.50 else 'di bawah optimal, mengindikasikan kerentanan akademik'}.</li>"
            )
            analisis_sks = (
                f"<li>Seluruh {total_sks_diambil} SKS berhasil diselesaikan tanpa beban mengulang.</li>"
                if total_sks_gagal == 0 else
                f"<li>Terdeteksi {total_sks_gagal} SKS tidak lulus — berisiko menghambat laju studi.</li>"
            )
            analisis_finansial = (
                "<li>Status UKT dan kewajiban finansial tertib dan lancar.</li>"
                if "Menunggak" not in ukt_lunas and tunggakan != "Ada Tunggakan" else
                "<li>Terdeteksi kendala tunggakan — berpotensi menjadi faktor penghambat studi.</li>"
            )
            disclaimer = (
                "<li>⚠ <i>Prediksi mempertimbangkan seluruh atribut profil mahasiswa. "
                "Keputusan klasifikasi murni berasal dari pola dataset pelatihan.</i></li>"
            )

            if prediction == 1:
                st.markdown(f"""
                <div class='result-success'>
                <div style='font-size:1.4rem; font-weight:700; color:#16a34a;'>Lulus Tepat Waktu</div>
                <div style='font-size:0.8rem; color:#4b5563; margin:4px 0 14px;'>
                    Tingkat Keyakinan Model: <b style='color:#16a34a;'>{prob_lulus:.1f}%</b></div>
                <div style='font-size:0.86rem; color:#111827; line-height:1.65;'>
                    <b>Insight Profil Analisis:</b>
                    <ul style='margin-top:6px; padding-left:20px; color:#4b5563;'>
                        {analisis_ips}{analisis_sks}{analisis_finansial}{disclaimer}
                    </ul>
                </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-danger'>
                <div style='font-size:1.4rem; font-weight:700; color:#dc2626;'>Berisiko Keterlambatan Studi</div>
                <div style='font-size:0.8rem; color:#4b5563; margin:4px 0 14px;'>
                    Tingkat Keyakinan Model: <b style='color:#dc2626;'>{prob_risiko:.1f}%</b></div>
                <div style='font-size:0.86rem; color:#111827; line-height:1.65;'>
                    <b>Insight Profil Analisis:</b>
                    <ul style='margin-top:6px; padding-left:20px; color:#4b5563;'>
                        {analisis_ips}{analisis_sks}{analisis_finansial}{disclaimer}
                    </ul>
                </div>
                </div>""", unsafe_allow_html=True)

        # ── Ringkasan input & catatan ─────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.72rem; color:#4b5563; font-weight:700; margin-bottom:8px;'>RINGKASAN INPUT DATA</div>", unsafe_allow_html=True)
        kv = {
            "SKS Lulus Sem 1": f"{s1_sks_lulus} SKS",
            "IPS Sem 1":        f"{s1_ips:.2f}",
            "SKS Lulus Sem 2": f"{s2_sks_lulus} SKS",
            "IPS Sem 2":        f"{s2_ips:.2f}",
            "Status UKT":      "Lunas" if "Menunggak" not in ukt_lunas else "Menunggak",
            "Beasiswa":        "Ya" if "Penerima" in beasiswa else "Tidak",
        }
        cols_kv = st.columns(3)
        for i, (k, v) in enumerate(kv.items()):
            with cols_kv[i % 3]:
                st.markdown(
                    f"<div class='kvbox'><div class='kvlbl'>{k}</div>"
                    f"<div class='kvval'>{v}</div></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("""
        <div class='note'>
        <b>Catatan Integritas Sistem:</b> Prediksi diproses oleh FastAPI backend menggunakan
        <code>predict_proba</code> model Machine Learning tanpa manipulasi pakar.
        Variabel makroekonomi menggunakan parameter <i>baseline dataset</i> untuk menjaga
        konsistensi ruang dimensi model. Bersifat <em>decision support</em> —
        bukan keputusan akademik final.
        </div>
        """, unsafe_allow_html=True)
