"""
=============================================================
STEP 3 - EXPLORATORY DATA ANALYSIS (EDA)
Input : 03_Hasil/02_data_siap_modeling.csv (dari Step 2)
Output: statistik deskriptif, visualisasi (.png), laporan outlier
=============================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------
# 0. KONFIGURASI
# -------------------------------------------------------------
INPUT_PATH = "03_Hasil/02_data_siap_modeling.csv"
OUTPUT_DIR = "03_Hasil"
PLOT_DIR = "03_Hasil/plots_eda"
REPORT_DIR = "laporan_hasil"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Klasifikasi curah hujan harian (kriteria BMKG, mm/hari)
BATAS_SEDANG = 20
BATAS_LEBAT = 50
BATAS_SANGAT_LEBAT = 100
BATAS_EKSTREM = 150

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
log("=" * 70)
log("STEP 3 - EXPLORATORY DATA ANALYSIS (EDA)")
log("=" * 70)
log(f"Waktu proses : {pd.Timestamp.now()}")
log(f"File input   : {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH, parse_dates=["TANGGAL"])
df["TAHUN"] = df["TANGGAL"].dt.year
df["BULAN"] = df["TANGGAL"].dt.month

log(f"\nShape data      : {df.shape}")
log(f"Jumlah stasiun  : {df['STATION_KEY'].nunique()}")
log(f"Rentang tanggal : {df['TANGGAL'].min().date()} s/d {df['TANGGAL'].max().date()}")


# -------------------------------------------------------------
# 2. STATISTIK DESKRIPTIF UMUM
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 3A - STATISTIK DESKRIPTIF UMUM (RAINFALL_MM, hasil imputasi)")
log("=" * 70)

desk = df["RAINFALL_MM"].describe()
log(desk.round(3).to_string())

log(f"\nHari tanpa hujan (0 mm)     : {(df['RAINFALL_MM']==0).sum():,} ({(df['RAINFALL_MM']==0).mean()*100:.2f}%)")
log(f"Hari hujan (>0 mm)          : {(df['RAINFALL_MM']>0).sum():,} ({(df['RAINFALL_MM']>0).mean()*100:.2f}%)")


# -------------------------------------------------------------
# 3. KLASIFIKASI INTENSITAS HUJAN (KRITERIA BMKG)
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 3B - KLASIFIKASI INTENSITAS HUJAN HARIAN (kriteria BMKG)")
log("=" * 70)

def klasifikasi(x):
    if x == 0:
        return "Tidak Hujan"
    elif x < BATAS_SEDANG:
        return "Ringan (0-20mm)"
    elif x < BATAS_LEBAT:
        return "Sedang (20-50mm)"
    elif x < BATAS_SANGAT_LEBAT:
        return "Lebat (50-100mm)"
    elif x < BATAS_EKSTREM:
        return "Sangat Lebat (100-150mm)"
    else:
        return "Ekstrem (>150mm)"

df["KATEGORI"] = df["RAINFALL_MM"].apply(klasifikasi)
kategori_count = df["KATEGORI"].value_counts()
kategori_pct = (df["KATEGORI"].value_counts(normalize=True) * 100).round(3)

log("\nJumlah hari per kategori (seluruh stasiun & tahun):")
for k in kategori_count.index:
    log(f"  {k:<28}: {kategori_count[k]:>8,} hari ({kategori_pct[k]:>6.3f}%)")

n_ekstrem = (df["RAINFALL_MM"] >= BATAS_EKSTREM).sum()
n_sangat_lebat_ke_atas = (df["RAINFALL_MM"] >= BATAS_SANGAT_LEBAT).sum()
log(f"\nTotal kejadian >=100mm (Sangat Lebat + Ekstrem): {n_sangat_lebat_ke_atas:,}")
log(f"Total kejadian >=150mm (Ekstrem)                : {n_ekstrem:,}")


# -------------------------------------------------------------
# 4. STATISTIK PER STASIUN
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 3C - STATISTIK PER STASIUN (Top 10 rata-rata tertinggi)")
log("=" * 70)

stat_stasiun = df.groupby(["STATION_KEY", "NAMA_STASIUN"]).agg(
    rata2_mm=("RAINFALL_MM", "mean"),
    median_mm=("RAINFALL_MM", "median"),
    max_mm=("RAINFALL_MM", "max"),
    std_mm=("RAINFALL_MM", "std"),
    hari_ekstrem=("RAINFALL_MM", lambda x: (x >= BATAS_EKSTREM).sum()),
).reset_index().sort_values("rata2_mm", ascending=False)

log(stat_stasiun.head(10).round(2).to_string(index=False))

stat_stasiun.to_csv(os.path.join(OUTPUT_DIR, "03_statistik_per_stasiun.csv"), index=False)


# -------------------------------------------------------------
# 5. DETEKSI OUTLIER (METODE IQR, PER STASIUN)
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 3D - DETEKSI OUTLIER (metode IQR per stasiun)")
log("=" * 70)

def deteksi_outlier_iqr(group):
    q1 = group["RAINFALL_MM"].quantile(0.25)
    q3 = group["RAINFALL_MM"].quantile(0.75)
    iqr = q3 - q1
    batas_atas = q3 + 1.5 * iqr
    return (group["RAINFALL_MM"] > batas_atas).sum(), batas_atas

outlier_summary = []
for (skey, name), group in df.groupby(["STATION_KEY", "NAMA_STASIUN"]):
    n_outlier, batas = deteksi_outlier_iqr(group)
    outlier_summary.append({
        "STATION_KEY": skey, "NAMA_STASIUN": name,
        "BATAS_ATAS_IQR_MM": round(batas, 2),
        "JUMLAH_OUTLIER": n_outlier,
        "PERSEN_OUTLIER": round(n_outlier / len(group) * 100, 3)
    })

outlier_df = pd.DataFrame(outlier_summary).sort_values("JUMLAH_OUTLIER", ascending=False)
log(f"\nTotal outlier terdeteksi (seluruh stasiun): {outlier_df['JUMLAH_OUTLIER'].sum():,}")
log(f"\nTop 10 stasiun dengan outlier terbanyak (kemungkinan rawan hujan ekstrem):")
log(outlier_df.head(10).to_string(index=False))

outlier_df.to_csv(os.path.join(OUTPUT_DIR, "03_deteksi_outlier_per_stasiun.csv"), index=False)

log("\n[CATATAN] Outlier IQR di data curah hujan TIDAK selalu berarti data error -")
log("bisa jadi itu justru kejadian hujan ekstrem asli yang penting untuk mitigasi bencana.")
log("Perlu pengecekan silang dengan RAINFALL_TETANGGA_AVG sebelum dianggap anomali sensor.")


# -------------------------------------------------------------
# 6. VISUALISASI
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 3E - VISUALISASI")
log("=" * 70)

# 6.1 Time series rata-rata regional harian
agg_harian = df.groupby("TANGGAL")["RAINFALL_MM"].mean()
fig, ax = plt.subplots(figsize=(14, 4))
agg_harian.rolling(30).mean().plot(ax=ax, color="steelblue")
ax.set_title("Rata-rata Curah Hujan Regional Sumbar (Rolling 30 Hari)")
ax.set_ylabel("mm/hari"); ax.set_xlabel("Tanggal")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_time_series_regional.png")
plt.close()
log("Saved: 01_time_series_regional.png")

# 6.2 Boxplot pola musiman per bulan
fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=df, x="BULAN", y="RAINFALL_MM", ax=ax, showfliers=False, color="lightblue")
ax.set_title("Pola Musiman Curah Hujan per Bulan (2015-2025)")
ax.set_ylabel("mm/hari"); ax.set_xlabel("Bulan")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/02_boxplot_musiman.png")
plt.close()
log("Saved: 02_boxplot_musiman.png")

# 6.3 Tren tahunan jumlah hari ekstrem
tren_ekstrem = df[df["RAINFALL_MM"] >= BATAS_EKSTREM].groupby("TAHUN").size()
fig, ax = plt.subplots(figsize=(10, 4))
tren_ekstrem.plot(kind="bar", ax=ax, color="indianred")
ax.set_title(f"Jumlah Kejadian Hujan Ekstrem (>={BATAS_EKSTREM}mm) per Tahun - Seluruh Stasiun")
ax.set_ylabel("Jumlah kejadian"); ax.set_xlabel("Tahun")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_tren_tahunan_ekstrem.png")
plt.close()
log("Saved: 03_tren_tahunan_ekstrem.png")

# 6.4 Peta sebaran stasiun (ukuran/warna = rata-rata curah hujan)
meta = df.drop_duplicates("STATION_KEY")[["STATION_KEY", "LATITUDE", "LONGITUDE"]]
peta_data = meta.merge(stat_stasiun[["STATION_KEY", "rata2_mm", "hari_ekstrem"]], on="STATION_KEY")
fig, ax = plt.subplots(figsize=(8, 9))
sc = ax.scatter(peta_data["LONGITUDE"], peta_data["LATITUDE"],
                 c=peta_data["hari_ekstrem"], s=80, cmap="Reds", edgecolor="black", linewidth=0.5)
plt.colorbar(sc, label="Jumlah hari ekstrem (>=150mm)")
ax.set_title("Sebaran 129 Pos Hujan Sumbar\n(warna = jumlah kejadian ekstrem historis)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/04_peta_sebaran_stasiun.png")
plt.close()
log("Saved: 04_peta_sebaran_stasiun.png")

# 6.5 Distribusi curah hujan (histogram, skala log utk visibilitas)
fig, ax = plt.subplots(figsize=(10, 4))
data_hujan = df[df["RAINFALL_MM"] > 0]["RAINFALL_MM"]
ax.hist(data_hujan, bins=80, color="teal", alpha=0.8)
ax.set_yscale("log")
ax.axvline(BATAS_EKSTREM, color="red", linestyle="--", label=f"Batas Ekstrem ({BATAS_EKSTREM}mm)")
ax.set_title("Distribusi Curah Hujan Harian (hari hujan saja, skala Y log)")
ax.set_xlabel("mm/hari"); ax.set_ylabel("Frekuensi (log)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/05_distribusi_curah_hujan.png")
plt.close()
log("Saved: 05_distribusi_curah_hujan.png")

log(f"\nSemua visualisasi tersimpan di folder: {PLOT_DIR}/")


# -------------------------------------------------------------
# 7. SIMPAN LAPORAN
# -------------------------------------------------------------
path_report_txt = os.path.join(REPORT_DIR, "03_laporan_eda.txt")
with open(path_report_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

log("\n" + "=" * 70)
log("OUTPUT TERSIMPAN")
log("=" * 70)
log(f"1. Statistik per stasiun    : 03_Hasil/03_statistik_per_stasiun.csv")
log(f"2. Deteksi outlier          : 03_Hasil/03_deteksi_outlier_per_stasiun.csv")
log(f"3. Visualisasi (5 file png) : {PLOT_DIR}/")
log(f"4. Laporan proses (txt)     : {path_report_txt}")

print("\nSELESAI.")