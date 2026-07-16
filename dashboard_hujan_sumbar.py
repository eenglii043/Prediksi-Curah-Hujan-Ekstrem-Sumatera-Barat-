"""
=============================================================
DASHBOARD SISTEM PERINGATAN DINI CURAH HUJAN EKSTREM SUMBAR (v2)
Tema: Langit Malam Berawan & Hujan
Perubahan v2:
- Kontras teks ditingkatkan
- Navigasi sidebar -> tab
- Fitur input curah hujan hari ini -> prediksi H+1..H+7 (analog historis)
=============================================================
CARA PAKAI:
1. Taruh file ini SATU FOLDER dengan folder 03_Hasil/
2. pip install streamlit plotly pandas
3. Jalankan: streamlit run dashboard_hujan_sumbar_v2.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import os
import joblib

# -------------------------------------------------------------
# 0. KONFIGURASI HALAMAN
# -------------------------------------------------------------
st.set_page_config(
    page_title="Peringatan Dini Curah Hujan Ekstrem - Sumbar",
    page_icon="🌧️",
    layout="wide",
)

DATA_DIR = "03_Hasil"
MODEL_DIR = "03_Hasil/models"
HORIZON_MAX = 7
FEATURE_COLS = [
    "LAG_1", "LAG_3", "LAG_7", "ROLL_MEAN_3", "ROLL_MEAN_7", "ROLL_MEAN_14",
    "TETANGGA_LAG1", "DOY_SIN", "DOY_COS", "BULAN_SIN", "BULAN_COS"
]

# Kode kabupaten/kota BPS Provinsi Sumatera Barat (2 digit setelah kode provinsi "13")
KODE_WILAYAH_SUMBAR = {
    "01": "Kab. Pesisir Selatan", "02": "Kab. Solok", "03": "Kab. Sijunjung",
    "04": "Kab. Tanah Datar", "05": "Kab. Padang Pariaman", "06": "Kab. Agam",
    "07": "Kab. Lima Puluh Kota", "08": "Kab. Pasaman", "09": "Kab. Solok Selatan",
    "10": "Kab. Dharmasraya", "11": "Kab. Pasaman Barat", "12": "Kab. Kepulauan Mentawai",
    "71": "Kota Padang", "72": "Kota Solok", "73": "Kota Sawahlunto",
    "74": "Kota Padang Panjang", "75": "Kota Bukittinggi", "76": "Kota Payakumbuh",
    "77": "Kota Pariaman",
}


def get_wilayah(station_key):
    """Ekstrak nama kabupaten/kota dari STATION_KEY berdasarkan kode BPS (format: 13 + kode kab/kota + ...)."""
    try:
        s = str(station_key)
        if s.startswith("13") and len(s) >= 4:
            kode = s[2:4]
            return KODE_WILAYAH_SUMBAR.get(kode, "Lainnya")
    except Exception:
        pass
    return "Lainnya"

# -------------------------------------------------------------
# 1. TEMA VISUAL (kontras ditingkatkan)
# -------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #090d1f 0%, #101636 35%, #182350 65%, #202f5c 100%);
    background-attachment: fixed;
}

.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(2px 2px at 20px 30px, #ffffff, transparent),
        radial-gradient(2px 2px at 90px 120px, #ffffff, transparent),
        radial-gradient(1.5px 1.5px at 160px 60px, #dbe4ff, transparent),
        radial-gradient(1.5px 1.5px at 230px 180px, #ffffff, transparent),
        radial-gradient(2px 2px at 300px 40px, #dbe4ff, transparent),
        radial-gradient(1.5px 1.5px at 340px 220px, #ffffff, transparent),
        radial-gradient(2px 2px at 400px 100px, #ffffff, transparent),
        radial-gradient(1.5px 1.5px at 60px 220px, #dbe4ff, transparent);
    background-repeat: repeat;
    background-size: 420px 260px;
    opacity: 0.5;
    animation: twinkle 4s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}

@keyframes twinkle {
    0%   { opacity: 0.3; }
    50%  { opacity: 0.65; }
    100% { opacity: 0.4; }
}

.cloud-layer {
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}
.cloud {
    position: absolute;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
    filter: blur(6px);
}
.cloud1 { width: 260px; height: 70px; top: 8%;  left: -20%; animation: drift 55s linear infinite; }
.cloud2 { width: 340px; height: 90px; top: 25%; left: -30%; animation: drift 75s linear infinite; animation-delay: -20s; }
.cloud3 { width: 200px; height: 60px; top: 55%; left: -25%; animation: drift 65s linear infinite; animation-delay: -40s; }

@keyframes drift {
    from { transform: translateX(0); }
    to   { transform: translateX(140vw); }
}

.rain {
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    background-image: repeating-linear-gradient(
        100deg,
        transparent 0px, transparent 38px,
        rgba(180,200,255,0.08) 39px, transparent 40px
    );
    animation: rainfall 0.6s linear infinite;
}
@keyframes rainfall {
    from { background-position: 0 0; }
    to   { background-position: -30px 60px; }
}

/* ===== KONTRAS TEKS DITINGKATKAN ===== */
h1, h2, h3, h4 {
    color: #ffffff !important;
    text-shadow: 0 0 12px rgba(120,150,255,0.25);
    font-weight: 600 !important;
}
p, span, label, li {
    color: #f0f3ff !important;
}
.stMarkdown, .stMarkdown p {
    color: #f0f3ff !important;
}
div[data-testid="stCaptionContainer"] {
    color: #c7d1f5 !important;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.09);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 14px;
    padding: 14px 18px;
    backdrop-filter: blur(6px);
}
div[data-testid="stMetric"] label {
    color: #d8e0ff !important;
    font-weight: 500 !important;
}
div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Tab navigasi */
button[data-baseweb="tab"] {
    background: rgba(255,255,255,0.06);
    border-radius: 10px 10px 0 0 !important;
    color: #d8e0ff !important;
    font-weight: 500 !important;
    padding: 10px 18px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(255,255,255,0.16) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-bottom: 3px solid #7dd3fc !important;
}
div[data-baseweb="tab-highlight"] { background-color: transparent !important; }
div[data-baseweb="tab-border"] { background-color: rgba(255,255,255,0.1) !important; }

.badge-pendek {
    background: linear-gradient(90deg, #2dd4bf, #22c1c3);
    color: #032c28; padding: 5px 14px; border-radius: 20px;
    font-weight: 700; font-size: 0.85rem; display: inline-block;
}
.badge-klimatologis {
    background: linear-gradient(90deg, #c4b5fd, #a5b4fc);
    color: #1c1240; padding: 5px 14px; border-radius: 20px;
    font-weight: 700; font-size: 0.85rem; display: inline-block;
}

.block-container {
    padding-top: 1.5rem;
    position: relative;
    z-index: 1;
}

div[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
}

/* Input widgets kontras */
.stSelectbox label, .stDateInput label, .stNumberInput label, .stRadio label {
    color: #ffffff !important;
    font-weight: 500 !important;
}
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.10) !important;
    color: #ffffff !important;
}
input {
    color: #ffffff !important;
}

/* ===== FIX KONTRAS: dropdown popup selectbox (daftar pilihan saat diklik) ===== */
div[data-baseweb="popover"] {
    background: #182350 !important;
}
div[data-baseweb="popover"] div[data-baseweb="menu"],
ul[data-baseweb="menu"] {
    background: #182350 !important;
}
li[role="option"] {
    background: #182350 !important;
    color: #ffffff !important;
}
li[role="option"]:hover, li[role="option"][aria-selected="true"] {
    background: rgba(255,255,255,0.18) !important;
    color: #ffffff !important;
}
div[data-baseweb="popover"] * {
    color: #ffffff !important;
}

/* ===== FIX KONTRAS: kotak input tanggal (date_input) yang tertutup ===== */
.stDateInput div[data-baseweb="input"],
div[data-testid="stDateInput"] div[data-baseweb="input"],
.stDateInput div[data-baseweb="base-input"],
div[data-testid="stDateInput"] div[data-baseweb="base-input"] {
    background-color: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
}
.stDateInput input,
div[data-testid="stDateInput"] input,
.stDateInput div[data-baseweb="input"] input,
div[data-testid="stDateInput"] div[data-baseweb="input"] input {
    background-color: transparent !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #ffffff !important;
}
.stDateInput svg, div[data-testid="stDateInput"] svg {
    fill: #ffffff !important;
}
div[data-baseweb="calendar"] {
    background: #182350 !important;
    color: #ffffff !important;
}
div[data-baseweb="calendar"] * {
    color: #ffffff !important;
}
div[data-baseweb="calendar"] button:hover {
    background: rgba(255,255,255,0.18) !important;
}

/* Alert boxes kontras */
div[data-testid="stAlert"] {
    background: rgba(255,255,255,0.10) !important;
}
div[data-testid="stAlert"] p {
    color: #ffffff !important;
    font-weight: 400 !important;
}

/* Panel info kustom */
.info-panel {
    background: rgba(255,255,255,0.08);
    border-left: 4px solid #7dd3fc;
    padding: 16px 20px;
    border-radius: 8px;
    margin: 10px 0;
}

/* ===== SISTEM ALARM MITIGASI BENCANA ===== */
.alarm-banner {
    border-radius: 12px;
    padding: 18px 24px;
    margin: 14px 0;
    font-weight: 700;
    font-size: 1.05rem;
    color: #ffffff !important;
    display: flex;
    align-items: center;
    gap: 12px;
}
.alarm-awas {
    background: linear-gradient(90deg, rgba(239,68,68,0.35), rgba(239,68,68,0.15));
    border: 2px solid #ef4444;
    animation: pulse-red 1.2s infinite;
}
.alarm-waspada {
    background: linear-gradient(90deg, rgba(245,158,11,0.30), rgba(245,158,11,0.12));
    border: 2px solid #f59e0b;
    animation: pulse-orange 1.8s infinite;
}
.alarm-siaga {
    background: linear-gradient(90deg, rgba(250,204,21,0.22), rgba(250,204,21,0.08));
    border: 2px solid #facc15;
}
.alarm-aman {
    background: rgba(34,197,94,0.15);
    border: 2px solid #22c55e;
}
@keyframes pulse-red {
    0%   { box-shadow: 0 0 0 0 rgba(239,68,68,0.5); }
    70%  { box-shadow: 0 0 0 14px rgba(239,68,68,0); }
    100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}
@keyframes pulse-orange {
    0%   { box-shadow: 0 0 0 0 rgba(245,158,11,0.4); }
    70%  { box-shadow: 0 0 0 10px rgba(245,158,11,0); }
    100% { box-shadow: 0 0 0 0 rgba(245,158,11,0); }
}
.alarm-icon {
    font-size: 1.8rem;
}

/* ===== AWAN LUCU & PETIR SESEKALI ===== */
.cloud-emoji {
    position: fixed;
    font-size: 2.6rem;
    opacity: 0.5;
    filter: drop-shadow(0 0 8px rgba(255,255,255,0.15));
    pointer-events: none;
    z-index: 0;
    animation: drift-emoji linear infinite;
}
.cloud-emoji.c1 { top: 6%;  left: -10%; animation-duration: 50s; }
.cloud-emoji.c2 { top: 18%; left: -15%; animation-duration: 65s; animation-delay: -15s; }
.cloud-emoji.c3 { top: 42%; left: -12%; animation-duration: 72s; animation-delay: -30s; }
.cloud-emoji.c4 { top: 62%; left: -18%; animation-duration: 58s; animation-delay: -8s; }
.cloud-emoji.c5 { top: 30%; left: -14%; animation-duration: 80s; animation-delay: -45s; }

@keyframes drift-emoji {
    from { transform: translateX(0) scale(1); }
    50%  { transform: translateX(65vw) scale(1.05); }
    to   { transform: translateX(130vw) scale(1); }
}

.bolt-emoji {
    position: fixed;
    font-size: 2.2rem;
    opacity: 0;
    pointer-events: none;
    z-index: 0;
    animation: flash-bolt ease-in-out infinite;
}
.bolt-emoji.b1 { top: 14%; left: 22%; animation-duration: 11s; animation-delay: 2s; }
.bolt-emoji.b2 { top: 48%; left: 72%; animation-duration: 14s; animation-delay: 6s; }
.bolt-emoji.b3 { top: 68%; left: 40%; animation-duration: 17s; animation-delay: 9s; }

@keyframes flash-bolt {
    0%, 90%, 100% { opacity: 0; transform: scale(1); }
    91%           { opacity: 1; transform: scale(1.15); }
    93%           { opacity: 0.2; transform: scale(1); }
    95%           { opacity: 0.9; transform: scale(1.1); }
    97%           { opacity: 0; }
}
</style>

<div class="cloud-layer">
    <div class="cloud cloud1">&nbsp;</div>
    <div class="cloud cloud2">&nbsp;</div>
    <div class="cloud cloud3">&nbsp;</div>
</div>
<div class="rain"></div>
<div class="cloud-emoji c1">☁️</div>
<div class="cloud-emoji c2">⛅</div>
<div class="cloud-emoji c3">☁️</div>
<div class="cloud-emoji c4">🌥️</div>
<div class="cloud-emoji c5">☁️</div>
<div class="bolt-emoji b1">⚡</div>
<div class="bolt-emoji b2">⚡</div>
<div class="bolt-emoji b3">⚡</div>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -------------------------------------------------------------
# 2. LOAD DATA (cached)
# -------------------------------------------------------------
@st.cache_data
def load_data():
    data = {}
    try:
        data["estimasi"] = pd.read_csv(
            os.path.join(DATA_DIR, "06_estimasi_risiko_historis_2026_2028.csv"),
            parse_dates=["TANGGAL"]
        )
    except FileNotFoundError:
        data["estimasi"] = None
    try:
        data["backtest"] = pd.read_csv(os.path.join(DATA_DIR, "06_backtest_akurasi_2tahun.csv"))
    except FileNotFoundError:
        data["backtest"] = None
    try:
        data["order"] = pd.read_csv(os.path.join(DATA_DIR, "06_sarima_order_terbaik_per_stasiun.csv"))
    except FileNotFoundError:
        data["order"] = None
    try:
        data["bersih"] = pd.read_csv(
            os.path.join(DATA_DIR, "02_data_siap_modeling.csv"), parse_dates=["TANGGAL"]
        )
    except FileNotFoundError:
        data["bersih"] = None
    try:
        data["xgb_pendek"] = pd.read_csv(
            os.path.join(DATA_DIR, "05_prediksi_jangka_pendek_terkini.csv"), parse_dates=["TANGGAL_PREDIKSI"]
        )
    except FileNotFoundError:
        data["xgb_pendek"] = None
    try:
        data["xgb_akurasi"] = pd.read_csv(os.path.join(DATA_DIR, "05_akurasi_xgboost_per_horizon.csv"))
    except FileNotFoundError:
        data["xgb_akurasi"] = None
    return data


@st.cache_resource
def load_xgboost_models():
    """Load 7 model XGBoost (H+1..H+7) sekali saja, disimpan di cache resource."""
    models = {}
    for h in range(1, HORIZON_MAX + 1):
        path = os.path.join(MODEL_DIR, f"xgb_horizon_h{h}.pkl")
        if os.path.exists(path):
            models[h] = joblib.load(path)
    return models

data = load_data()
estimasi_df = data["estimasi"]
backtest_df = data["backtest"]
order_df = data["order"]
historis_df = data["bersih"]
xgb_pendek_df = data["xgb_pendek"]
xgb_akurasi_df = data["xgb_akurasi"]

if estimasi_df is None:
    st.error(
        "⚠️ File `06_estimasi_risiko_historis_2026_2028.csv` tidak ditemukan di folder `03_Hasil/`. "
        "Pastikan file dashboard ini ditaruh satu folder dengan folder `03_Hasil/`."
    )
    st.stop()

estimasi_df["WILAYAH"] = estimasi_df["STATION_KEY"].apply(get_wilayah)
wilayah_list = sorted(estimasi_df["WILAYAH"].unique().tolist())
station_list = sorted(estimasi_df["NAMA_STASIUN"].dropna().unique().tolist())

station_wilayah_map = estimasi_df.drop_duplicates("STATION_KEY")[["STATION_KEY", "NAMA_STASIUN", "WILAYAH"]]


def pilih_stasiun_bertingkat(key_prefix, label_wilayah="Pilih Kabupaten/Kota", label_stasiun="Pilih Pos Hujan"):
    """
    Widget 2 tingkat: pilih wilayah (kabupaten/kota) dulu, baru pilih pos hujan
    di dalam wilayah itu. Memudahkan orang awam yang tidak familiar dengan
    nama-nama pos hujan/daerah kecil.
    Return: nama stasiun terpilih (str) atau None kalau tidak ada data.
    """
    col1, col2 = st.columns(2)
    with col1:
        wilayah_pilih = st.selectbox(label_wilayah, wilayah_list, key=f"{key_prefix}_wilayah")
    stasiun_di_wilayah = sorted(
        station_wilayah_map[station_wilayah_map["WILAYAH"] == wilayah_pilih]["NAMA_STASIUN"].tolist()
    )
    with col2:
        if len(stasiun_di_wilayah) == 0:
            st.selectbox(label_stasiun, ["(tidak ada pos hujan)"], key=f"{key_prefix}_stasiun_kosong", disabled=True)
            return None
        stasiun_pilih = st.selectbox(label_stasiun, stasiun_di_wilayah, key=f"{key_prefix}_stasiun")
    return stasiun_pilih


# -------------------------------------------------------------
# FUNGSI: PREDIKSI XGBOOST GENUINE (input 14 hari terakhir -> H+1..H+7)
# -------------------------------------------------------------
def prediksi_xgboost_dari_input(station_key, curah_14_hari, tanggal_input_terakhir):
    """
    curah_14_hari: list 14 nilai curah hujan, index 0 = 14 hari lalu, index 13 = hari ini (H).
    Membangun fitur lag/rolling/kalender persis seperti saat training, lalu
    memanggil model XGBoost asli (bukan analog historis) untuk tiap horizon.
    """
    models = load_xgboost_models()
    if len(models) == 0:
        return None, "Model XGBoost tidak ditemukan di folder 03_Hasil/models/. Jalankan dulu 05_train_xgboost_pendek.py."

    arr = np.array(curah_14_hari, dtype=float)  # arr[13] = hari ini (H), arr[0] = H-13

    lag_1 = arr[13]
    lag_3 = arr[11]
    lag_7 = arr[7]
    roll_mean_3 = arr[11:14].mean()
    roll_mean_7 = arr[7:14].mean()
    roll_mean_14 = arr[0:14].mean()

    # Fitur tetangga: tidak tersedia dari input manual (butuh data live stasiun sekitar),
    # jadi didekati dari rata-rata historis rasio tetangga/curah hujan stasiun ini.
    tetangga_lag1 = lag_1
    if historis_df is not None:
        sub = historis_df[historis_df["STATION_KEY"] == station_key]
        if len(sub) > 0 and "RAINFALL_TETANGGA_AVG" in sub.columns and sub["RAINFALL_MM"].mean() > 0:
            rasio = (sub["RAINFALL_TETANGGA_AVG"] / sub["RAINFALL_MM"].replace(0, np.nan)).median()
            if pd.notna(rasio):
                tetangga_lag1 = lag_1 * rasio

    doy = tanggal_input_terakhir.timetuple().tm_yday
    bulan = tanggal_input_terakhir.month
    doy_sin, doy_cos = np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)
    bulan_sin, bulan_cos = np.sin(2*np.pi*bulan/12), np.cos(2*np.pi*bulan/12)

    fitur = pd.DataFrame([{
        "LAG_1": lag_1, "LAG_3": lag_3, "LAG_7": lag_7,
        "ROLL_MEAN_3": roll_mean_3, "ROLL_MEAN_7": roll_mean_7, "ROLL_MEAN_14": roll_mean_14,
        "TETANGGA_LAG1": tetangga_lag1,
        "DOY_SIN": doy_sin, "DOY_COS": doy_cos, "BULAN_SIN": bulan_sin, "BULAN_COS": bulan_cos,
    }])[FEATURE_COLS]

    rows = []
    for h in range(1, HORIZON_MAX + 1):
        if h not in models:
            continue
        pred = max(0.0, float(models[h].predict(fitur)[0]))
        rows.append({
            "H": f"H+{h}",
            "TANGGAL": tanggal_input_terakhir + pd.Timedelta(days=h),
            "PREDIKSI_MM": pred,
        })

    if len(rows) == 0:
        return None, "Tidak ada model horizon yang berhasil dimuat."

    return pd.DataFrame(rows), None


# -------------------------------------------------------------
# FUNGSI: SISTEM ALARM MITIGASI BENCANA
# -------------------------------------------------------------

ALARM_SOUND_B64 = "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"


def kategori_risiko(nilai_mm):
    """
    Kategori risiko sederhana untuk mitigasi bencana hidrometeorologi,
    berdasarkan ambang curah hujan harian (mm). Threshold merujuk pada
    pola umum kriteria BMKG untuk curah hujan ekstrem.
    """
    if nilai_mm >= 150:
        return "AWAS", "🔴", "alarm-awas"
    elif nilai_mm >= 100:
        return "WASPADA", "🟠", "alarm-waspada"
    elif nilai_mm >= 50:
        return "SIAGA", "🟡", "alarm-siaga"
    else:
        return "AMAN", "🟢", "alarm-aman"


PESAN_MITIGASI = {
    "AWAS": "Curah hujan sangat tinggi. Berpotensi banjir/longsor. Siapkan evakuasi, jauhi lereng & bantaran sungai, pantau arahan BPBD setempat.",
    "WASPADA": "Curah hujan tinggi. Waspadai genangan & aliran sungai. Siapkan rencana evakuasi & pantau kondisi sekitar.",
    "SIAGA": "Curah hujan cukup tinggi. Tetap waspada, pantau perkembangan cuaca berikutnya.",
    "AMAN": "Curah hujan dalam rentang normal. Tetap pantau informasi BMKG secara berkala.",
}


def render_alarm_banner(nilai_mm, konteks="", tampilkan_suara=False):
    """Render banner alarm visual (+ opsional bunyi sirine) berdasarkan nilai curah hujan."""
    level, ikon, css_class = kategori_risiko(nilai_mm)
    pesan = PESAN_MITIGASI[level]
    st.markdown(
        f'''<div class="alarm-banner {css_class}">
        <span class="alarm-icon">{ikon}</span>
        <div><b>STATUS: {level}</b> {konteks} — Estimasi {nilai_mm:.0f} mm<br>
        <span style="font-weight:400; font-size:0.92rem;">{pesan}</span></div>
        </div>''',
        unsafe_allow_html=True
    )
    if tampilkan_suara and level in ("AWAS", "WASPADA"):
        import streamlit.components.v1 as components
        components.html(
            f'''<audio autoplay><source src="data:audio/wav;base64,{ALARM_SOUND_B64}" type="audio/wav"></audio>''',
            height=0
        )


def tampilkan_skenario_curah_hujan(mean_val, p50, p90, p95):
    """
    Tampilkan estimasi curah hujan dalam bahasa awam (bukan istilah statistik P50/P90/P95),
    supaya dipahami masyarakat umum, BMKG, maupun BPBD tanpa perlu latar belakang statistik.
    Nilai teknis tetap disertakan lewat tooltip (help) bagi yang butuh detail teknis.
    """
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "☔ Kondisi Biasanya",
            f"{p50:.0f} mm",
            help=(
                "Ini kondisi yang PALING SERING terjadi berdasarkan pola 11 tahun terakhir "
                "(istilah teknis: median/P50). Anggap ini sebagai 'perkiraan paling mungkin'."
            )
        )
    with c2:
        st.metric(
            "🌧️ Jika Lebih Deras dari Biasa",
            f"{p90:.0f} mm",
            help=(
                "Ada sekitar 1 dari 10 kemungkinan curah hujan mencapai angka ini atau lebih tinggi "
                "(istilah teknis: P90). Ini level yang perlu diwaspadai."
            )
        )
    with c3:
        st.metric(
            "⛈️ Skenario Terburuk (Jarang Terjadi)",
            f"{p95:.0f} mm",
            help=(
                "Ada sekitar 1 dari 20 kemungkinan (jarang, tapi bisa terjadi) curah hujan mencapai "
                "angka ini atau lebih ekstrem (istilah teknis: P95). Ini skenario ekstrem."
            )
        )
    st.caption(f"📊 Rata-rata perhitungan model: {mean_val:.1f} mm")


# -------------------------------------------------------------
# 3. HEADER
# -------------------------------------------------------------
st.markdown(
    """
    <h1>🌩️ Sistem Peringatan Dini Curah Hujan Ekstrem</h1>
    <p style="font-size:1.05rem;">
    Provinsi Sumatera Barat &nbsp;•&nbsp; 129 Pos Hujan &nbsp;•&nbsp; Model Hybrid SARIMA-Klimatologi Harmonik
    </p>
    <span class="badge-pendek">🟢 Prediksi Jangka Pendek (H+1–H+7)</span>&nbsp;&nbsp;
    <span class="badge-klimatologis">🟣 Estimasi Risiko Historis (2026–2028)</span>
    """,
    unsafe_allow_html=True
)
st.write("")

with st.expander("📖 **Cara Membaca Dashboard Ini** (klik untuk membuka panduan singkat)", expanded=False):
    st.markdown(
        """
        **Untuk siapa dashboard ini?** Masyarakat umum, BMKG, dan BPBD di Sumatera Barat — semua fitur
        dijelaskan dengan bahasa sederhana, tanpa perlu latar belakang statistik.

        #### 🏷️ Arti Badge Warna
        - 🟢 **Prediksi Jangka Pendek (H+1–H+7)** = prediksi asli dari model machine learning (XGBoost),
          akurat untuk 7 hari ke depan dari data terakhir.
        - 🟣 **Estimasi Risiko Historis (2026–2028)** = BUKAN prediksi cuaca presisi, melainkan gambaran
          kecenderungan berdasarkan pola 11 tahun terakhir. Semakin jauh tanggalnya, semakin tidak pasti.

        #### 🚨 Arti Level Alarm
        - 🟢 **AMAN** — curah hujan normal, tetap pantau info BMKG berkala
        - 🟡 **SIAGA** — curah hujan cukup tinggi, mulai waspada
        - 🟠 **WASPADA** — curah hujan tinggi, siapkan rencana evakuasi
        - 🔴 **AWAS** — curah hujan sangat tinggi, berpotensi banjir/longsor, segera siapkan evakuasi

        #### 🗂️ Isi Tiap Tab
        - **🏠 Beranda** — ringkasan alarm seluruh Sumbar & peta sebaran 129 pos hujan
        - **🌦️ Prediksi Jangka Pendek** — lihat prediksi 7 hari ke depan per pos hujan, atau input data
          14 hari terakhir untuk dapat prediksi langsung dari model
        - **🔮 Estimasi Tanggal Tertentu** — cek kecenderungan curah hujan di tanggal tertentu (2026-2028)
        - **📊 Akurasi Model** — transparansi seberapa akurat model ini (untuk BMKG/BPBD menilai reliabilitas)
        - **ℹ️ Tentang & Metodologi** — penjelasan teknis lengkap untuk yang ingin tahu detail ilmiahnya
        """
    )
st.write("")

# -------------------------------------------------------------
# 4. NAVIGASI VIA TAB
# -------------------------------------------------------------
tab_beranda, tab_pendek, tab_klimatologis, tab_akurasi, tab_tentang = st.tabs(
    ["🏠 Beranda", "🌦️ Prediksi Jangka Pendek", "🔮 Estimasi Tanggal Tertentu",
     "📊 Akurasi Model", "ℹ️ Tentang & Metodologi"]
)


# -------------------------------------------------------------
# TAB: BERANDA
# -------------------------------------------------------------
with tab_beranda:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Pos Hujan", f"{estimasi_df['STATION_KEY'].nunique()}")
    col2.metric("Rentang Data Historis", "2015 – 2025")
    col3.metric("Cakupan Estimasi", "2026 – 2028")
    if backtest_df is not None and len(backtest_df) > 0:
        col4.metric("MAE Rata-rata (Backtest)", f"{backtest_df['MAE_mm_bulanan'].mean():.2f} mm/bulan")
    else:
        col4.metric("MAE Rata-rata (Backtest)", "N/A")

    st.markdown("### 📅 Pilih Tanggal untuk Lihat Kondisi")
    peta_tanggal = st.date_input(
        "Ringkasan alarm & peta di bawah akan menampilkan kondisi pada tanggal ini:",
        value=estimasi_df["TANGGAL"].min().date(),
        min_value=estimasi_df["TANGGAL"].min().date(),
        max_value=estimasi_df["TANGGAL"].max().date(),
        key="peta_tgl"
    )
    df_peta = estimasi_df[estimasi_df["TANGGAL"] == pd.Timestamp(peta_tanggal)]

    st.markdown(f"### 🚨 Ringkasan Alarm Mitigasi Bencana — {pd.Timestamp(peta_tanggal).strftime('%d %B %Y')}")
    if len(df_peta) > 0:
        nilai_p90 = df_peta["ESTIMASI_P90"]
        n_awas = (nilai_p90 >= 150).sum()
        n_waspada = ((nilai_p90 >= 100) & (nilai_p90 < 150)).sum()
        n_siaga = ((nilai_p90 >= 50) & (nilai_p90 < 100)).sum()
        n_aman = (nilai_p90 < 50).sum()

        if n_awas > 0:
            render_alarm_banner(nilai_p90.max(), konteks=f"— {n_awas} pos hujan berstatus AWAS pada tanggal ini")
        elif n_waspada > 0:
            render_alarm_banner(nilai_p90[nilai_p90 >= 100].max(), konteks=f"— {n_waspada} pos hujan berstatus WASPADA pada tanggal ini")
        elif n_siaga > 0:
            render_alarm_banner(nilai_p90[nilai_p90 >= 50].max(), konteks=f"— {n_siaga} pos hujan berstatus SIAGA pada tanggal ini")
        else:
            st.markdown(
                '<div class="alarm-banner alarm-aman"><span class="alarm-icon">🟢</span>'
                '<div><b>STATUS REGIONAL: AMAN</b><br>'
                '<span style="font-weight:400; font-size:0.92rem;">Tidak ada pos hujan dengan potensi curah hujan ekstrem pada tanggal ini.</span></div></div>',
                unsafe_allow_html=True
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 AWAS", int(n_awas))
        c2.metric("🟠 WASPADA", int(n_waspada))
        c3.metric("🟡 SIAGA", int(n_siaga))
        c4.metric("🟢 AMAN", int(n_aman))
        metode_ringkasan = df_peta["METODE"].iloc[0]
        if metode_ringkasan == "PREDIKSI_JANGKA_PENDEK":
            st.caption("🟢 Ringkasan ini berdasarkan prediksi genuine model XGBoost (H+1–H+7).")
        else:
            st.caption("🟣 Ringkasan ini berdasarkan estimasi kecenderungan historis, bukan prediksi cuaca presisi.")
    else:
        st.info("Data tidak tersedia untuk tanggal ini.")

    st.markdown("### 🗺️ Peta Sebaran Pos Hujan")

    if len(df_peta) > 0:
        fig_map = px.scatter_mapbox(
            df_peta,
            lat="LATITUDE", lon="LONGITUDE",
            color="ESTIMASI_P90",
            size="ESTIMASI_P90",
            size_max=22,
            hover_name="NAMA_STASIUN",
            hover_data={
                "ESTIMASI_MEAN": ":.1f", "ESTIMASI_P90": ":.1f",
                "METODE": True, "LATITUDE": False, "LONGITUDE": False
            },
            labels={
                "ESTIMASI_P90": "Potensi Hujan Lebat (mm)",
                "ESTIMASI_MEAN": "Perkiraan Rata-rata (mm)",
                "METODE": "Metode",
            },
            color_continuous_scale="Turbo",
            zoom=6.3,
            center={"lat": -0.9, "lon": 100.4},
            mapbox_style="carto-darkmatter",
            height=520,
        )
        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            font_color="#ffffff",
            coloraxis_colorbar=dict(title="Potensi<br>Hujan Lebat<br>(mm)"),
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption(
            "💡 Warna & ukuran titik menunjukkan potensi hujan lebat (skenario 'lebih deras dari biasa'). "
            "Semakin merah/besar, semakin tinggi potensinya."
        )
    else:
        st.info("Tidak ada data untuk tanggal tersebut.")

    st.markdown("### 🏆 Top 10 Pos Hujan dengan Risiko Tertinggi (7 Hari ke Depan)")
    tgl_awal = estimasi_df["TANGGAL"].min()
    df_7hari = estimasi_df[estimasi_df["TANGGAL"] <= tgl_awal + pd.Timedelta(days=6)]
    top10 = (
        df_7hari.groupby("NAMA_STASIUN")["ESTIMASI_P90"]
        .mean().sort_values(ascending=False).head(10).reset_index()
    )
    fig_bar = px.bar(
        top10, x="ESTIMASI_P90", y="NAMA_STASIUN", orientation="h",
        color="ESTIMASI_P90", color_continuous_scale="Blues",
        labels={"ESTIMASI_P90": "Potensi Hujan Lebat (mm)", "NAMA_STASIUN": ""},
    )
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff", yaxis=dict(autorange="reversed"),
        height=420,
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# -------------------------------------------------------------
# TAB: PREDIKSI JANGKA PENDEK
# -------------------------------------------------------------
with tab_pendek:
    st.markdown("## 🟢 Prediksi Jangka Pendek (H+1 s/d H+7)")

    sub_tab_model, sub_tab_input = st.tabs(["📈 Lihat Prediksi Model", "✍️ Input Curah Hujan Hari Ini"])

    with sub_tab_model:
        st.markdown(
            '<div class="info-panel">Prediksi berikut dihasilkan oleh model <b>XGBoost</b> (bukan SARIMA) untuk 7 hari '
            'ke depan sejak data historis terakhir tersedia (31 Desember 2025), menggunakan fitur lag curah hujan, '
            'rata-rata bergerak, curah hujan tetangga terdekat, dan pola kalender musiman.</div>',
            unsafe_allow_html=True
        )
        if xgb_pendek_df is None:
            st.warning(
                "File `05_prediksi_jangka_pendek_terkini.csv` tidak ditemukan. "
                "Jalankan dulu `05_train_xgboost_pendek.py` untuk menghasilkan prediksi ini."
            )
        else:
            stasiun_pilih = pilih_stasiun_bertingkat("model")
            df_pendek = xgb_pendek_df[xgb_pendek_df["NAMA_STASIUN"] == stasiun_pilih].sort_values("HORIZON")

            if len(df_pendek) == 0:
                st.warning("Data prediksi XGBoost tidak tersedia untuk stasiun ini.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[f"H+{h}" for h in df_pendek["HORIZON"]], y=df_pendek["PREDIKSI_XGBOOST_MM"],
                    marker_color="#2dd4bf", text=df_pendek["PREDIKSI_XGBOOST_MM"].round(1), textposition="outside",
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#ffffff", height=420,
                    yaxis_title="Prediksi Curah Hujan (mm)", xaxis_title="Horizon",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Tabel Detail")
                tabel_tampil = df_pendek[["TANGGAL_PREDIKSI", "HORIZON", "PREDIKSI_XGBOOST_MM"]].copy()
                tabel_tampil.columns = ["Tanggal", "Horizon (hari)", "Prediksi XGBoost (mm)"]
                st.dataframe(tabel_tampil.round(1), use_container_width=True, hide_index=True)

                maks_model = df_pendek["PREDIKSI_XGBOOST_MM"].max()
                hari_maks_model = df_pendek.loc[df_pendek["PREDIKSI_XGBOOST_MM"].idxmax(), "HORIZON"]
                render_alarm_banner(maks_model, konteks=f"— puncak diperkirakan H+{hari_maks_model}")

                if xgb_akurasi_df is not None:
                    with st.expander("📐 Lihat akurasi model (MAE/RMSE) per horizon"):
                        st.dataframe(xgb_akurasi_df, use_container_width=True, hide_index=True)

    with sub_tab_input:
        st.markdown(
            '<div class="info-panel"><b>Cara kerja:</b> Masukkan curah hujan <b>14 hari terakhir</b> di stasiun '
            'pilihan Anda (hari ini dan 13 hari sebelumnya). Dashboard akan menghitung fitur lag & rata-rata bergerak, '
            'lalu memanggil <b>model XGBoost asli</b> yang sudah dilatih (bukan analog historis) untuk memprediksi '
            'H+1 s/d H+7 secara genuine.</div>',
            unsafe_allow_html=True
        )

        models_loaded = load_xgboost_models()
        if len(models_loaded) == 0:
            st.warning(
                "Model XGBoost tidak ditemukan di folder `03_Hasil/models/`. "
                "Jalankan dulu `05_train_xgboost_pendek.py` untuk melatih dan menyimpan modelnya."
            )
        else:
            stasiun_input = pilih_stasiun_bertingkat("input_manual")
            col_tgl, _ = st.columns([1, 1])
            with col_tgl:
                tanggal_input = st.date_input("Tanggal hari ini (H)", value=date.today(), key="tgl_input_manual")

            st.markdown("##### 📝 Isi curah hujan 14 hari terakhir (mm)")
            st.caption("Baris paling bawah = hari ini (H). Ubah angkanya sesuai data lapangan.")

            default_tabel = pd.DataFrame({
                "Hari": [f"H-{13-i}" if i < 13 else "H (hari ini)" for i in range(14)],
                "Curah Hujan (mm)": [0.0] * 14,
            })
            tabel_input = st.data_editor(
                default_tabel, use_container_width=True, hide_index=True,
                disabled=["Hari"], key="editor_14hari",
                column_config={"Curah Hujan (mm)": st.column_config.NumberColumn(min_value=0.0, max_value=500.0, step=1.0)}
            )

            tombol_prediksi = st.button("🔍 Prediksi 7 Hari ke Depan (XGBoost)", use_container_width=True)
            aktifkan_suara = st.checkbox("🔊 Aktifkan bunyi alarm jika status WASPADA/AWAS", value=True)

            if tombol_prediksi:
                station_key_input = estimasi_df[estimasi_df["NAMA_STASIUN"] == stasiun_input]["STATION_KEY"].iloc[0]
                curah_list = tabel_input["Curah Hujan (mm)"].tolist()

                df_hasil, err = prediksi_xgboost_dari_input(station_key_input, curah_list, pd.Timestamp(tanggal_input))

                if err:
                    st.error(err)
                else:
                    st.success(f"Prediksi berhasil dihitung menggunakan model XGBoost untuk stasiun {stasiun_input}.")

                    fig4 = go.Figure()
                    fig4.add_trace(go.Bar(
                        x=df_hasil["H"], y=df_hasil["PREDIKSI_MM"],
                        marker_color="#2dd4bf", name="Prediksi XGBoost",
                        text=df_hasil["PREDIKSI_MM"].round(1), textposition="outside",
                    ))
                    fig4.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#ffffff", height=400,
                        yaxis_title="Prediksi Curah Hujan (mm)", xaxis_title="Hari ke depan",
                    )
                    st.plotly_chart(fig4, use_container_width=True)

                    tabel_hasil = df_hasil.copy()
                    tabel_hasil.columns = ["Horizon", "Tanggal", "Prediksi (mm)"]
                    st.dataframe(tabel_hasil.round(1), use_container_width=True, hide_index=True)

                    maks_pred = df_hasil["PREDIKSI_MM"].max()
                    hari_maks = df_hasil.loc[df_hasil["PREDIKSI_MM"].idxmax(), "H"]
                    render_alarm_banner(maks_pred, konteks=f"— puncak diperkirakan {hari_maks}", tampilkan_suara=aktifkan_suara)


# -------------------------------------------------------------
# TAB: ESTIMASI TANGGAL TERTENTU (klimatologis)
# -------------------------------------------------------------
with tab_klimatologis:
    st.markdown("## 🟣 Estimasi Risiko Berbasis Pola Historis (2026–2028)")
    st.markdown(
        '<div class="info-panel">⚠️ <b>Penting:</b> Ini <b>bukan prediksi cuaca presisi</b>. Untuk tanggal yang jauh ke depan, '
        'tidak ada model ilmiah yang bisa memprediksi curah hujan harian secara akurat — bahkan model NWP canggih BMKG '
        'sekalipun hanya reliable untuk ~7–14 hari ke depan. Nilai di bawah ini adalah <b>estimasi probabilitas berdasarkan '
        'pola historis 11 tahun (2015–2025)</b>, dikombinasikan dengan tren musiman SARIMA. Gunakan sebagai gambaran '
        'kecenderungan, bukan kepastian.</div>',
        unsafe_allow_html=True
    )

    stasiun_pilih2 = pilih_stasiun_bertingkat("klimatologis")
    col_tgl2, _ = st.columns([1, 1])
    with col_tgl2:
        tgl_pilih = st.date_input(
            "Pilih tanggal (2026–2028)",
            value=date(2027, 1, 15),
            min_value=estimasi_df["TANGGAL"].min().date(),
            max_value=estimasi_df["TANGGAL"].max().date(),
            key="tgl_klim"
        )

    baris = estimasi_df[
        (estimasi_df["NAMA_STASIUN"] == stasiun_pilih2) &
        (estimasi_df["TANGGAL"] == pd.Timestamp(tgl_pilih))
    ]

    if len(baris) == 0:
        st.info("Data tidak ditemukan untuk kombinasi stasiun/tanggal ini.")
    else:
        row = baris.iloc[0]
        metode_label = "🟢 Prediksi Jangka Pendek" if row["METODE"] == "PREDIKSI_JANGKA_PENDEK" else "🟣 Estimasi Risiko Historis"
        st.markdown(f"**Metode:** {metode_label}  |  **Tingkat Ketidakpastian:** {row['TINGKAT_KETIDAKPASTIAN']*100:.0f}%")

        tampilkan_skenario_curah_hujan(row["ESTIMASI_MEAN"], row["ESTIMASI_P50"], row["ESTIMASI_P90"], row["ESTIMASI_P95"])

        render_alarm_banner(row["ESTIMASI_P90"], konteks="(berdasar skenario 'lebih deras dari biasa' — kecenderungan historis, bukan kepastian)")

        st.markdown("#### 📅 Konteks Bulan Terkait (klimatologi historis)")
        bulan_terkait = estimasi_df[
            (estimasi_df["NAMA_STASIUN"] == stasiun_pilih2) &
            (estimasi_df["TANGGAL"].dt.month == pd.Timestamp(tgl_pilih).month) &
            (estimasi_df["TANGGAL"].dt.year == pd.Timestamp(tgl_pilih).year)
        ]
        if len(bulan_terkait) > 0:
            fig2 = px.line(
                bulan_terkait, x="TANGGAL", y="ESTIMASI_P50",
                labels={"ESTIMASI_P50": "Estimasi P50 (mm)", "TANGGAL": "Tanggal"},
            )
            fig2.update_traces(line_color="#a5b4fc", line_width=3)
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ffffff", height=350,
            )
            st.plotly_chart(fig2, use_container_width=True)


# -------------------------------------------------------------
# TAB: AKURASI MODEL
# -------------------------------------------------------------
with tab_akurasi:
    st.markdown("## 📊 Transparansi Akurasi Model (Backtest 2 Tahun Terakhir)")
    st.markdown(
        '<div class="info-panel">Halaman ini menunjukkan seberapa akurat model SARIMA memprediksi curah hujan bulanan, '
        'diuji dengan menyembunyikan 2 tahun data terakhir (2024–2025) lalu membandingkan hasil model dengan data aktual.</div>',
        unsafe_allow_html=True
    )

    if backtest_df is None or len(backtest_df) == 0:
        st.warning("Data backtest tidak ditemukan.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("MAE Rata-rata (semua stasiun)", f"{backtest_df['MAE_mm_bulanan'].mean():.2f} mm/bulan")
        col2.metric("RMSE Rata-rata (semua stasiun)", f"{backtest_df['RMSE_mm_bulanan'].mean():.2f} mm/bulan")

        st.markdown("#### Distribusi MAE per Stasiun")
        fig3 = px.histogram(
            backtest_df, x="MAE_mm_bulanan", nbins=30,
            color_discrete_sequence=["#a5b4fc"],
            labels={"MAE_mm_bulanan": "MAE (mm/bulan)"},
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ffffff", height=350,
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### Tabel Akurasi per Stasiun")
        merge_meta = estimasi_df.drop_duplicates("STATION_KEY")[["STATION_KEY", "NAMA_STASIUN"]]
        tabel_acc = backtest_df.merge(merge_meta, on="STATION_KEY", how="left")
        tabel_acc = tabel_acc[["NAMA_STASIUN", "MAE_mm_bulanan", "RMSE_mm_bulanan"]].sort_values("MAE_mm_bulanan")
        tabel_acc.columns = ["Nama Stasiun", "MAE (mm/bulan)", "RMSE (mm/bulan)"]
        st.dataframe(tabel_acc, use_container_width=True, hide_index=True, height=400)

        if order_df is not None:
            with st.expander("🔧 Lihat detail order SARIMA per stasiun (teknis)"):
                st.dataframe(order_df, use_container_width=True, hide_index=True)


# -------------------------------------------------------------
# TAB: TENTANG & METODOLOGI
# -------------------------------------------------------------
with tab_tentang:
    st.markdown("## ℹ️ Tentang Dashboard Ini")
    st.markdown(
        """
        Dashboard ini dikembangkan untuk mendukung **mitigasi bencana hidrometeorologi** di Provinsi
        Sumatera Barat, menggunakan data curah hujan harian dari **129 pos hujan** selama periode
        **2015–2025**.

        ### 🧪 Metodologi
        **1. Prediksi Jangka Pendek (H+1 – H+7) — dari data historis terakhir**
        Menggunakan model **XGBoost** (regresi, 7 model terpisah per horizon), dilatih regional dari
        seluruh 129 stasiun sekaligus. Fitur yang dipakai: lag curah hujan (1, 3, 7 hari), rata-rata
        bergerak (3, 7, 14 hari), curah hujan tetangga terdekat (lag 1 hari), dan pola kalender musiman
        (bulan & hari-dalam-tahun). Divalidasi dengan hold-out 2 tahun terakhir (2024–2025).

        **2. Prediksi Jangka Pendek — dari input 14 hari curah hujan terakhir**
        Menggunakan **model XGBoost yang sama persis** dengan poin 1 (bukan pendekatan analog/pencarian
        pola serupa) — Anda memasukkan data 14 hari terakhir, dashboard menghitung fitur lag & rolling
        yang dibutuhkan, lalu model asli memprediksi H+1 s/d H+7. Fitur tetangga didekati dari rasio
        historis stasiun tersebut karena data real-time tetangga tidak tersedia lewat input manual.

        **3. Estimasi Risiko Historis (2026–2028)**
        Untuk tanggal yang jauh ke depan, dashboard **tidak** menyajikan "prediksi cuaca" — karena
        secara ilmiah tidak ada model curah hujan harian yang akurat untuk horizon waktu sejauh itu,
        termasuk model NWP profesional BMKG sekalipun (yang reliable ~7–14 hari).

        Sebagai gantinya, digunakan pendekatan **probabilitas klimatologis**:
        - **Klimatologi harmonik**: pola musiman historis per stasiun dimodelkan dengan regresi Fourier
          (mean & persentil P50/P75/P90/P95 tiap hari-dalam-tahun)
        - **SARIMA musiman bulanan**: menangkap tren jangka menengah/panjang per stasiun, dipilih otomatis
          via stepwise search berbasis AIC
        - **Blending**: proyeksi bulanan SARIMA dipakai sebagai faktor skala terhadap kurva klimatologi harian

        ### ✅ Validasi Akurasi
        Model diuji dengan **backtest**: 2 tahun data terakhir (2024–2025) disembunyikan, model dilatih
        hanya dengan data sebelumnya, lalu forecast dibandingkan dengan data aktual. Hasil metrik akurasi
        dapat dilihat di tab **📊 Akurasi Model**.

        ### ⚠️ Batasan Penting
        - Estimasi untuk 2026–2028 adalah **kecenderungan berbasis pola historis**, bukan prediksi presisi
        - Semakin jauh tanggal dari hari ini, semakin tinggi tingkat ketidakpastian (ditampilkan eksplisit)
        - Untuk keputusan operasional mitigasi bencana, tetap gunakan **prakiraan resmi BMKG** sebagai
          rujukan utama — dashboard ini bersifat pelengkap dan alat bantu kesadaran risiko

        ### 📊 Sumber Data
        Data curah hujan harian 129 pos hujan Sumatera Barat, periode 2015–2025, diolah melalui proses
        cleaning (penanganan sentinel value, penggabungan stasiun terduplikasi, seleksi kelengkapan data),
        imputasi (interpolasi + KNN), dan feature engineering spasial (fitur tetangga terdekat).
        """
    )

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; opacity:0.75; padding:10px 0 24px 0; font-size:0.85rem;">
    Dashboard ini dikembangkan oleh <b>Enggli Rahmadhani</b> (NIM: 123450043)<br>
    Laporan Kerja Praktik &nbsp;•&nbsp; Program Studi Sains Data &nbsp;•&nbsp; Institut Teknologi Sumatera &nbsp;•&nbsp; 2026
    </div>
    """,
    unsafe_allow_html=True
)
