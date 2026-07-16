"""
=============================================================
STEP 4 - FEATURE ENGINEERING
Input : 03_Hasil/02_data_siap_modeling.csv (dari Step 2)
Output: dataset final siap SARIMA-XGBoost (fitur lag/rolling/spasial + target H1-H7 + label ekstrem)
=============================================================
"""

import pandas as pd
import numpy as np
import os

# -------------------------------------------------------------
# 0. KONFIGURASI
# -------------------------------------------------------------
INPUT_PATH = "03_Hasil/02_data_siap_modeling.csv"
OUTPUT_DIR = "03_Hasil"
REPORT_DIR = "laporan_hasil"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

HORIZON_MAX = 7          # prediksi H+1 s/d H+7
BATAS_EKSTREM_UTAMA = 100   # mm, label biner utama
BATAS_EKSTREM_INFO = 150    # mm, label tambahan (kriteria BMKG ekstrem)
LAG_LIST = [1, 2, 3, 7, 14, 30]
ROLL_WINDOWS = [3, 7, 14, 30]

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
log("=" * 70)
log("STEP 4 - FEATURE ENGINEERING")
log("=" * 70)
log(f"Waktu proses : {pd.Timestamp.now()}")
log(f"File input   : {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH, parse_dates=["TANGGAL"])
df = df.sort_values(["STATION_KEY", "TANGGAL"]).reset_index(drop=True)
log(f"\nShape data awal : {df.shape}")
log(f"Jumlah stasiun   : {df['STATION_KEY'].nunique()}")


# -------------------------------------------------------------
# 2. FITUR WAKTU (SIKLIKAL)
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 4A - FITUR WAKTU (bulan, hari-dalam-tahun, encoding siklikal)")
log("=" * 70)

df["BULAN"] = df["TANGGAL"].dt.month
df["HARI_TAHUN"] = df["TANGGAL"].dt.dayofyear
df["SIN_BULAN"] = np.sin(2 * np.pi * df["BULAN"] / 12)
df["COS_BULAN"] = np.cos(2 * np.pi * df["BULAN"] / 12)
df["SIN_HARI_TAHUN"] = np.sin(2 * np.pi * df["HARI_TAHUN"] / 365.25)
df["COS_HARI_TAHUN"] = np.cos(2 * np.pi * df["HARI_TAHUN"] / 365.25)

log("Fitur dibuat: BULAN, HARI_TAHUN, SIN_BULAN, COS_BULAN, SIN_HARI_TAHUN, COS_HARI_TAHUN")


# -------------------------------------------------------------
# 3. FITUR LAG (PER STASIUN, TIDAK BOCOR ANTAR STASIUN)
# -------------------------------------------------------------
log("\n" + "=" * 70)
log(f"STEP 4B - FITUR LAG: {LAG_LIST} hari")
log("=" * 70)

grp = df.groupby("STATION_KEY", group_keys=False)
for lag in LAG_LIST:
    df[f"LAG_{lag}"] = grp["RAINFALL_MM"].shift(lag)
log(f"Fitur lag dibuat: {[f'LAG_{l}' for l in LAG_LIST]}")


# -------------------------------------------------------------
# 4. FITUR ROLLING (MEAN, MAX, SUM) - PAKAI DATA MASA LALU SAJA (shift 1 dulu)
# -------------------------------------------------------------
log("\n" + "=" * 70)
log(f"STEP 4C - FITUR ROLLING: window {ROLL_WINDOWS} hari (mean/max/sum)")
log("=" * 70)

df["_RAIN_SHIFT1"] = grp["RAINFALL_MM"].shift(1)
grp_shift = df.groupby("STATION_KEY", group_keys=False)["_RAIN_SHIFT1"]

for w in ROLL_WINDOWS:
    df[f"ROLL_MEAN_{w}"] = grp_shift.transform(lambda s: s.rolling(w, min_periods=1).mean())
    df[f"ROLL_MAX_{w}"] = grp_shift.transform(lambda s: s.rolling(w, min_periods=1).max())
    df[f"ROLL_SUM_{w}"] = grp_shift.transform(lambda s: s.rolling(w, min_periods=1).sum())

df = df.drop(columns=["_RAIN_SHIFT1"])
log(f"Fitur rolling dibuat untuk window: {ROLL_WINDOWS} (mean, max, sum) -> {len(ROLL_WINDOWS)*3} kolom")


# -------------------------------------------------------------
# 5. FITUR HARI HUJAN BERTURUT-TURUT (CONSEC_RAIN)
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 4D - FITUR CONSEC_RAIN (hari hujan berturut-turut, s/d kemarin)")
log("=" * 70)

def hitung_consec_rain(rain_series):
    """Jumlah hari hujan (>0mm) berturut-turut hingga hari sebelumnya."""
    is_rain = (rain_series.shift(1) > 0).astype(int)
    consec = is_rain.groupby((is_rain != is_rain.shift()).cumsum()).cumsum()
    consec = consec.where(is_rain == 1, 0)
    return consec

df["CONSEC_RAIN"] = df.groupby("STATION_KEY", group_keys=False)["RAINFALL_MM"].apply(hitung_consec_rain)
log("Fitur CONSEC_RAIN dibuat (jumlah hari hujan berturut-turut sampai H-1).")


# -------------------------------------------------------------
# 6. FITUR SPASIAL (TETANGGA) - LAG & ROLLING
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 4E - FITUR SPASIAL TETANGGA (lag & rolling dari RAINFALL_TETANGGA_AVG)")
log("=" * 70)

grp_tetangga = df.groupby("STATION_KEY", group_keys=False)["RAINFALL_TETANGGA_AVG"]
df["TETANGGA_LAG1"] = grp_tetangga.shift(1)
df["_TETANGGA_SHIFT1"] = grp_tetangga.shift(1)
df["TETANGGA_ROLL_MEAN_7"] = df.groupby("STATION_KEY", group_keys=False)["_TETANGGA_SHIFT1"].transform(
    lambda s: s.rolling(7, min_periods=1).mean()
)
df = df.drop(columns=["_TETANGGA_SHIFT1"])

df["SPASIAL_X_ROLL7"] = df["TETANGGA_ROLL_MEAN_7"] * df["ROLL_MEAN_7"]

log("Fitur dibuat: TETANGGA_LAG1, TETANGGA_ROLL_MEAN_7, SPASIAL_X_ROLL7")


# -------------------------------------------------------------
# 7. TARGET MULTI-HORIZON (H+1 s/d H+7) - REGRESI
# -------------------------------------------------------------
log("\n" + "=" * 70)
log(f"STEP 4F - TARGET REGRESI H+1 s/d H+{HORIZON_MAX} (nilai curah hujan masa depan)")
log("=" * 70)

grp_target = df.groupby("STATION_KEY", group_keys=False)["RAINFALL_MM"]
for h in range(1, HORIZON_MAX + 1):
    df[f"TARGET_H{h}"] = grp_target.shift(-h)

log(f"Target dibuat: {[f'TARGET_H{h}' for h in range(1, HORIZON_MAX+1)]}")


# -------------------------------------------------------------
# 8. LABEL KLASIFIKASI EKSTREM (MAX 7 HARI KE DEPAN >= THRESHOLD)
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 4G - LABEL KLASIFIKASI EKSTREM (maksimum H+1 s/d H+7)")
log("=" * 70)

target_cols = [f"TARGET_H{h}" for h in range(1, HORIZON_MAX + 1)]
df["MAX_RAINFALL_7HARI_KEDEPAN"] = df[target_cols].max(axis=1)

df["LABEL_EKSTREM_100_7HARI"] = (df["MAX_RAINFALL_7HARI_KEDEPAN"] >= BATAS_EKSTREM_UTAMA).astype("Int64")
df["LABEL_EKSTREM_150_7HARI"] = (df["MAX_RAINFALL_7HARI_KEDEPAN"] >= BATAS_EKSTREM_INFO).astype("Int64")

mask_incomplete = df["MAX_RAINFALL_7HARI_KEDEPAN"].isna()
df.loc[mask_incomplete, ["LABEL_EKSTREM_100_7HARI", "LABEL_EKSTREM_150_7HARI"]] = pd.NA

n_label100 = (df["LABEL_EKSTREM_100_7HARI"] == 1).sum()
n_label150 = (df["LABEL_EKSTREM_150_7HARI"] == 1).sum()
n_valid_label = df["LABEL_EKSTREM_100_7HARI"].notna().sum()

log(f"LABEL_EKSTREM_100_7HARI (label utama) : {n_label100:,} positif dari {n_valid_label:,} baris valid "
    f"({n_label100/n_valid_label*100:.3f}%)")
log(f"LABEL_EKSTREM_150_7HARI (info tambahan): {n_label150:,} positif dari {n_valid_label:,} baris valid "
    f"({n_label150/n_valid_label*100:.3f}%)")


# -------------------------------------------------------------
# 9. BUANG BARIS DENGAN LAG/TARGET TIDAK LENGKAP (UJUNG DERET PER STASIUN)
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 4H - BUANG BARIS DENGAN FITUR LAG TIDAK LENGKAP (awal deret)")
log("=" * 70)

lag_cols = [f"LAG_{l}" for l in LAG_LIST]
n_sebelum = len(df)
df_model = df.dropna(subset=lag_cols).copy()
n_setelah = len(df_model)
log(f"Baris sebelum buang (lag tidak lengkap di awal deret tiap stasiun): {n_sebelum:,}")
log(f"Baris setelah buang                                              : {n_setelah:,}")
log(f"Baris terbuang (30 hari pertama tiap 129 stasiun, wajar)         : {n_sebelum - n_setelah:,}")

log(f"\n[CATATAN] Baris dengan TARGET_H1..H7 / LABEL kosong (7 hari terakhir tiap stasiun)")
log("TETAP DISIMPAN dalam dataset (untuk keperluan prediksi real-time / dashboard),")
log("tapi HARUS di-drop saat training model (gunakan .dropna(subset=['LABEL_EKSTREM_100_7HARI']) sebelum fit).")


# -------------------------------------------------------------
# 10. SIMPAN OUTPUT
# -------------------------------------------------------------
path_data_final = os.path.join(OUTPUT_DIR, "04_data_siap_modeling_final.csv")
path_report_txt = os.path.join(REPORT_DIR, "04_laporan_feature_engineering.txt")

df_model.to_csv(path_data_final, index=False)

log("\n" + "=" * 70)
log("RINGKASAN KOLOM DATASET FINAL")
log("=" * 70)
log(f"Total kolom: {df_model.shape[1]}")
log(f"Total baris: {df_model.shape[0]:,}")
log(f"\nDaftar kolom:")
for c in df_model.columns:
    log(f"  - {c}")

with open(path_report_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

log("\n" + "=" * 70)
log("OUTPUT TERSIMPAN")
log("=" * 70)
log(f"1. Dataset final siap modeling : {path_data_final}")
log(f"2. Laporan proses (txt)        : {path_report_txt}")

print("\nSELESAI.")