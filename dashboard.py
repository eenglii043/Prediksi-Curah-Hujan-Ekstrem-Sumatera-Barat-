"""
=============================================================
DASHBOARD PERINGATAN DINI CURAH HUJAN EKSTREM - SUMATERA BARAT
Untuk: BMKG, BPBD, dan masyarakat umum
Model: Hybrid SARIMA-XGBoost (129 pos hujan, prediksi 7 hari ke depan)
=============================================================
Cara jalankan: streamlit run dashboard_hujan_ekstrem_sumbar.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# -------------------------------------------------------------
# 0. KONFIGURASI HALAMAN
# -------------------------------------------------------------
st.set_page_config(
    page_title="Peringatan Dini Curah Hujan Ekstrem - Sumbar",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "03_Hasil/05_data_dengan_fitur_sarima.csv"
MODEL_PATH = "03_Hasil/model/xgboost_model_final.pkl"
METADATA_PATH = "03_Hasil/model/metadata.pkl"

# batas kategori risiko (berdasar probabilitas model)
BATAS_RISIKO_RENDAH = 0.30
BATAS_RISIKO_TINGGI = 0.65  # mendekati threshold hasil tuning model


# -------------------------------------------------------------
# 1. LOAD DATA & MODEL (DI-CACHE SUPAYA CEPAT)
# -------------------------------------------------------------
@st.cache_data(show_spinner="Memuat data curah hujan...")
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["TANGGAL"])
    return df

@st.cache_resource(show_spinner="Memuat model prediksi...")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return model, metadata

df = load_data()
model, metadata = load_model()
fitur_cols = metadata["fitur_cols"]
threshold_model = metadata.get("threshold_terbaik", 0.5)


def kategori_risiko(prob):
    if prob >= BATAS_RISIKO_TINGGI:
        return "Tinggi"
    elif prob >= BATAS_RISIKO_RENDAH:
        return "Sedang"
    else:
        return "Rendah"

WARNA_RISIKO = {"Tinggi": "#d62728", "Sedang": "#ff9800", "Rendah": "#2ca02c"}


# -------------------------------------------------------------
# 2. HITUNG PREDIKSI "KONDISI TERKINI" (SIMULASI, DATA HISTORIS TERAKHIR)
# -------------------------------------------------------------
@st.cache_data(show_spinner="Menghitung prediksi risiko seluruh stasiun...")
def hitung_prediksi_terkini(_df, _fitur_cols):
    tanggal_terkini = _df["TANGGAL"].max()
    df_terkini = _df[_df["TANGGAL"] == tanggal_terkini].dropna(subset=_fitur_cols).copy()

    if df_terkini.empty:
        # fallback: cari tanggal terakhir yang datanya lengkap
        df_valid = _df.dropna(subset=_fitur_cols)
        tanggal_terkini = df_valid["TANGGAL"].max()
        df_terkini = _df[_df["TANGGAL"] == tanggal_terkini].dropna(subset=_fitur_cols).copy()

    X = df_terkini[_fitur_cols]
    df_terkini["PROB_EKSTREM"] = model.predict_proba(X)[:, 1]
    df_terkini["KATEGORI_RISIKO"] = df_terkini["PROB_EKSTREM"].apply(kategori_risiko)
    return df_terkini, tanggal_terkini

df_prediksi, TANGGAL_TERKINI = hitung_prediksi_terkini(df, fitur_cols)


# -------------------------------------------------------------
# 3. SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.title("🌧️ Info Sistem")
    st.markdown(f"""
    **Model:** Hybrid SARIMA-XGBoost
    **Cakupan:** {df['STATION_KEY'].nunique()} pos hujan, Sumatera Barat
    **Horizon prediksi:** 7 hari ke depan
    **Data per:** {TANGGAL_TERKINI.strftime('%d %B %Y')}
    """)
    st.divider()
    st.markdown("""
    **Kategori Risiko** (peluang hujan ≥100mm dalam 7 hari ke depan):
    - 🟢 **Rendah**: < 30%
    - 🟠 **Sedang**: 30% – 65%
    - 🔴 **Tinggi**: ≥ 65%
    """)
    st.divider()
    st.caption(
        "⚠️ **Mode Demo**: dashboard ini memakai data historis untuk simulasi, "
        "bukan data live. Untuk operasional nyata, sumber data perlu dihubungkan "
        "ke feed BMKG real-time."
    )


# -------------------------------------------------------------
# 4. HEADER UTAMA
# -------------------------------------------------------------
st.title("🌧️ Sistem Peringatan Dini Curah Hujan Ekstrem — Sumatera Barat")
st.markdown(
    "Prediksi risiko hujan ekstrem (≥100mm dalam 7 hari ke depan) untuk mitigasi "
    "bencana hidrometeorologi, memakai model **Hybrid SARIMA-XGBoost** pada "
    f"{df['STATION_KEY'].nunique()} pos hujan di seluruh Sumatera Barat."
)

col1, col2, col3, col4 = st.columns(4)
n_tinggi = (df_prediksi["KATEGORI_RISIKO"] == "Tinggi").sum()
n_sedang = (df_prediksi["KATEGORI_RISIKO"] == "Sedang").sum()
n_rendah = (df_prediksi["KATEGORI_RISIKO"] == "Rendah").sum()
col1.metric("Total Pos Hujan", f"{len(df_prediksi)}")
col2.metric("🔴 Risiko Tinggi", f"{n_tinggi}")
col3.metric("🟠 Risiko Sedang", f"{n_sedang}")
col4.metric("🟢 Risiko Rendah", f"{n_rendah}")

st.divider()


# -------------------------------------------------------------
# 5. TAB NAVIGASI
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🚨 Peringatan Dini", "📍 Prediksi Detail Stasiun", "📊 Riwayat & Statistik", "ℹ️ Tentang"
])


# =========================================================================
# TAB 1: PERINGATAN DINI (PETA UTAMA)
# =========================================================================
with tab1:
    st.subheader("Peta Sebaran Risiko Hujan Ekstrem")
    st.caption("Klik titik pada peta untuk melihat detail stasiun, atau pilih dari daftar di bawah.")

    fig_peta = px.scatter_mapbox(
        df_prediksi,
        lat="LATITUDE", lon="LONGITUDE",
        color="KATEGORI_RISIKO",
        color_discrete_map=WARNA_RISIKO,
        category_orders={"KATEGORI_RISIKO": ["Rendah", "Sedang", "Tinggi"]},
        hover_name="NAMA_STASIUN",
        hover_data={"PROB_EKSTREM": ":.1%", "LATITUDE": False, "LONGITUDE": False, "KATEGORI_RISIKO": True},
        size=[12] * len(df_prediksi),
        zoom=6.3,
        height=520,
        mapbox_style="open-street-map",
    )
    fig_peta.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, legend_title_text="Kategori Risiko")

    event = st.plotly_chart(fig_peta, use_container_width=True, on_select="rerun", key="peta_utama")

    stasiun_terpilih_peta = None
    if event and event.get("selection") and event["selection"].get("points"):
        idx_terpilih = event["selection"]["points"][0]["point_index"]
        stasiun_terpilih_peta = df_prediksi.iloc[idx_terpilih]["STATION_KEY"]

    st.divider()

    col_kiri, col_kanan = st.columns([1, 1.3])

    with col_kiri:
        st.markdown("#### 🔴 Daftar Stasiun Risiko Tinggi & Sedang")
        df_perhatian = df_prediksi[df_prediksi["KATEGORI_RISIKO"].isin(["Tinggi", "Sedang"])].sort_values(
            "PROB_EKSTREM", ascending=False
        )
        if df_perhatian.empty:
            st.success("Tidak ada stasiun berisiko sedang/tinggi saat ini.")
        else:
            tampil = df_perhatian[["NAMA_STASIUN", "KATEGORI_RISIKO", "PROB_EKSTREM"]].copy()
            tampil["PROB_EKSTREM"] = (tampil["PROB_EKSTREM"] * 100).round(1).astype(str) + "%"
            tampil.columns = ["Nama Stasiun", "Kategori Risiko", "Peluang Ekstrem"]
            st.dataframe(tampil, hide_index=True, use_container_width=True, height=350)

    with col_kanan:
        pilihan_default = stasiun_terpilih_peta if stasiun_terpilih_peta else df_prediksi.iloc[0]["STATION_KEY"]
        daftar_stasiun = df_prediksi.set_index("STATION_KEY")["NAMA_STASIUN"].to_dict()
        stasiun_dipilih = st.selectbox(
            "Pilih / cek detail stasiun:",
            options=list(daftar_stasiun.keys()),
            format_func=lambda x: daftar_stasiun[x],
            index=list(daftar_stasiun.keys()).index(pilihan_default),
            key="pilih_stasiun_tab1",
        )

        baris = df_prediksi[df_prediksi["STATION_KEY"] == stasiun_dipilih].iloc[0]
        kat = baris["KATEGORI_RISIKO"]
        prob = baris["PROB_EKSTREM"]

        st.markdown(f"### {baris['NAMA_STASIUN']}")
        warna = WARNA_RISIKO[kat]
        st.markdown(
            f"<div style='padding:16px;border-radius:10px;background-color:{warna}22;"
            f"border-left:6px solid {warna};'>"
            f"<b style='font-size:20px;color:{warna};'>Risiko {kat}</b><br>"
            f"Peluang hujan ≥100mm dalam 7 hari ke depan: <b>{prob*100:.1f}%</b>"
            f"</div>", unsafe_allow_html=True
        )

        if kat == "Tinggi":
            st.error("🚨 **Rekomendasi:** Siaga penuh. Koordinasikan kesiapsiagaan dengan BPBD setempat, "
                      "waspada potensi banjir/longsor di area rawan.")
        elif kat == "Sedang":
            st.warning("⚠️ **Rekomendasi:** Pantau perkembangan cuaca 2-3 hari ke depan, siapkan jalur evakuasi.")
        else:
            st.success("✅ **Rekomendasi:** Kondisi normal, tetap pantau info cuaca berkala.")

        st.caption(f"Curah hujan tercatat terakhir ({TANGGAL_TERKINI.strftime('%d %b %Y')}): "
                    f"{baris['RAINFALL_MM']:.1f} mm")


# =========================================================================
# TAB 2: PREDIKSI DETAIL STASIUN
# =========================================================================
with tab2:
    st.subheader("Prediksi Detail per Stasiun")

    daftar_stasiun2 = df_prediksi.set_index("STATION_KEY")["NAMA_STASIUN"].to_dict()
    stasiun2 = st.selectbox(
        "Pilih stasiun:", options=list(daftar_stasiun2.keys()),
        format_func=lambda x: daftar_stasiun2[x], key="pilih_stasiun_tab2"
    )

    baris2 = df_prediksi[df_prediksi["STATION_KEY"] == stasiun2].iloc[0]
    prob2 = baris2["PROB_EKSTREM"]
    kat2 = baris2["KATEGORI_RISIKO"]

    colA, colB = st.columns([1, 2])

    with colA:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob2 * 100,
            number={"suffix": "%"},
            title={"text": "Peluang Hujan Ekstrem (7 hari)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": WARNA_RISIKO[kat2]},
                "steps": [
                    {"range": [0, 30], "color": "#e8f5e9"},
                    {"range": [30, 65], "color": "#fff3e0"},
                    {"range": [65, 100], "color": "#ffebee"},
                ],
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.metric("Elevasi", f"{baris2['ELEVASI_M']:.0f} m")
        st.metric("Curah hujan terakhir tercatat", f"{baris2['RAINFALL_MM']:.1f} mm")
        st.metric("Rata-rata curah hujan tetangga", f"{baris2['RAINFALL_TETANGGA_AVG']:.1f} mm")

    with colB:
        st.markdown("#### Riwayat Curah Hujan 90 Hari Terakhir")
        df_stasiun_hist = df[
            (df["STATION_KEY"] == stasiun2) & (df["TANGGAL"] >= TANGGAL_TERKINI - timedelta(days=90))
        ].sort_values("TANGGAL")

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=df_stasiun_hist["TANGGAL"], y=df_stasiun_hist["RAINFALL_MM"],
            name="Curah Hujan (mm)", marker_color="steelblue"
        ))
        fig_hist.add_hline(y=100, line_dash="dash", line_color="red",
                             annotation_text="Batas Ekstrem (100mm)")
        fig_hist.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20),
                                 yaxis_title="mm/hari", xaxis_title="Tanggal")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### Riwayat Kejadian Ekstrem (≥100mm) - Seluruh Periode Data")
        kejadian_ekstrem = df[(df["STATION_KEY"] == stasiun2) & (df["RAINFALL_MM"] >= 100)].sort_values(
            "TANGGAL", ascending=False
        )[["TANGGAL", "RAINFALL_MM"]]
        if kejadian_ekstrem.empty:
            st.info("Tidak ada kejadian hujan ≥100mm tercatat di stasiun ini.")
        else:
            kejadian_ekstrem["TANGGAL"] = kejadian_ekstrem["TANGGAL"].dt.strftime("%d %b %Y")
            kejadian_ekstrem.columns = ["Tanggal", "Curah Hujan (mm)"]
            st.dataframe(kejadian_ekstrem, hide_index=True, use_container_width=True, height=200)


# =========================================================================
# TAB 3: RIWAYAT & STATISTIK REGIONAL
# =========================================================================
with tab3:
    st.subheader("Riwayat & Statistik Regional Sumatera Barat")

    col_a, col_b = st.columns(2)
    with col_a:
        rentang_tahun = st.slider(
            "Rentang tahun:", int(df["TANGGAL"].dt.year.min()), int(df["TANGGAL"].dt.year.max()),
            (int(df["TANGGAL"].dt.year.min()), int(df["TANGGAL"].dt.year.max()))
        )
    with col_b:
        filter_stasiun = st.multiselect(
            "Filter stasiun (kosongkan = semua):",
            options=list(df_prediksi.set_index("STATION_KEY")["NAMA_STASIUN"].to_dict().values()),
        )

    df_filtered = df[(df["TANGGAL"].dt.year >= rentang_tahun[0]) & (df["TANGGAL"].dt.year <= rentang_tahun[1])]
    if filter_stasiun:
        df_filtered = df_filtered[df_filtered["NAMA_STASIUN"].isin(filter_stasiun)]

    st.markdown("#### Tren Curah Hujan Regional (Rata-rata Harian, Rolling 30 Hari)")
    agg_harian = df_filtered.groupby("TANGGAL")["RAINFALL_MM"].mean().rolling(30).mean()
    fig_tren = px.line(agg_harian, height=350)
    fig_tren.update_layout(showlegend=False, xaxis_title="Tanggal", yaxis_title="mm/hari")
    st.plotly_chart(fig_tren, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### Pola Musiman per Bulan")
        df_filtered_copy = df_filtered.copy()
        df_filtered_copy["BULAN_NAMA"] = df_filtered_copy["TANGGAL"].dt.strftime("%b")
        urutan_bulan = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig_box = px.box(df_filtered_copy, x="BULAN_NAMA", y="RAINFALL_MM",
                           category_orders={"BULAN_NAMA": urutan_bulan}, height=350)
        fig_box.update_layout(xaxis_title="Bulan", yaxis_title="mm/hari")
        st.plotly_chart(fig_box, use_container_width=True)

    with col_d:
        st.markdown("#### Top 10 Stasiun Terrawan Kejadian Ekstrem")
        top10 = df_filtered[df_filtered["RAINFALL_MM"] >= 100].groupby("NAMA_STASIUN").size().sort_values(
            ascending=False
        ).head(10)
        fig_top10 = px.bar(top10, orientation="h", height=350)
        fig_top10.update_layout(showlegend=False, xaxis_title="Jumlah kejadian ≥100mm", yaxis_title="")
        fig_top10.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_top10, use_container_width=True)

    st.markdown("#### Statistik Ringkas")
    col_e, col_f, col_g, col_h = st.columns(4)
    col_e.metric("Total Hari Data", f"{df_filtered['TANGGAL'].nunique():,}")
    col_f.metric("Rata-rata Curah Hujan", f"{df_filtered['RAINFALL_MM'].mean():.2f} mm")
    col_g.metric("Curah Hujan Maksimum", f"{df_filtered['RAINFALL_MM'].max():.1f} mm")
    col_h.metric("Total Kejadian Ekstrem", f"{(df_filtered['RAINFALL_MM']>=100).sum():,}")


# =========================================================================
# TAB 4: TENTANG
# =========================================================================
with tab4:
    st.subheader("Tentang Sistem Ini")
    st.markdown(f"""
    ### 🎯 Tujuan
    Dashboard ini dikembangkan untuk mendukung **mitigasi bencana hidrometeorologi**
    di Sumatera Barat, dengan menyediakan prediksi risiko hujan ekstrem 7 hari ke
    depan bagi **BMKG**, **BPBD**, dan **masyarakat umum**.

    ### 📊 Data
    - Sumber: data pos hujan historis **2015-2025** ({df['STATION_KEY'].nunique()} pos hujan aktif
      setelah proses cleaning, dari total 164 pos hujan mentah)
    - Proses: cleaning (penanganan sentinel value, penggabungan stasiun), imputasi
      (interpolasi linear + KNN spasial), normalisasi, dan feature engineering
      (lag, rolling statistics, fitur spasial antar-stasiun)

    ### 🤖 Model
    - **SARIMA**: menangkap komponen tren & musiman curah hujan regional (agregasi bulanan)
    - **XGBoost**: klasifikasi biner risiko ekstrem (curah hujan ≥100mm dalam 7 hari ke depan),
      memakai fitur lag/rolling/spasial + komponen SARIMA
    - Validasi: TimeSeriesSplit (5 fold, berbasis tanggal, bukan random split)
    - Hyperparameter dioptimasi dengan **Optuna**

    ### ⚠️ Keterbatasan
    - Model ini memprediksi berdasarkan **pola curah hujan historis**, belum memasukkan
      data atmosfer real-time (kelembapan, tekanan udara, citra satelit/radar cuaca)
    - Recall model saat ini masih terbatas — **tidak semua kejadian ekstrem
      terdeteksi**. Dashboard ini adalah alat bantu pendukung keputusan,
      **bukan pengganti** analisis dan keputusan resmi BMKG/BPBD
    - Dashboard versi ini memakai **data historis untuk simulasi/demo**, belum
      terhubung ke feed data live

    ### 📞 Kontak & Sumber
    Dikembangkan sebagai bagian dari kerja praktik / penelitian mahasiswa Data
    Science, Institut Teknologi Sumatera (ITERA), bekerja sama dengan
    Stasiun Meteorologi Kelas 2 Padang Pariaman Minangkabau.
    """)