"""
=============================================================
STEP 2 - IMPUTASI MISSING VALUE, NORMALISASI, & FITUR TETANGGA
Input : 03_Hasil/01_data_bersih_curah_hujan_sumbar.csv (dari Step 1)
Output: data siap-modeling + laporan imputasi
=============================================================
"""

import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------------
# 0. KONFIGURASI
# -------------------------------------------------------------
INPUT_PATH = "03_Hasil/01_data_bersih_curah_hujan_sumbar.csv"
OUTPUT_DIR = "03_Hasil"
REPORT_DIR = "laporan_hasil"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

GAP_INTERPOLASI_MAX = 7   # hari; gap <= ini diinterpolasi linear
K_NEIGHBORS_IMPUTE = 5    # jumlah stasiun tetangga dipakai KNNImputer
K_NEIGHBORS_FITUR = 5     # jumlah tetangga untuk fitur RAINFALL_TETANGGA_AVG

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
log("=" * 70)
log("STEP 2 - IMPUTASI, NORMALISASI, FITUR TETANGGA")
log("=" * 70)
log(f"Waktu proses : {pd.Timestamp.now()}")
log(f"File input   : {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH, parse_dates=["TANGGAL"])
log(f"\nShape data awal   : {df.shape}")
log(f"Jumlah stasiun    : {df['STATION_KEY'].nunique()}")
log(f"Missing value awal: {df['RAINFALL_MM'].isna().sum():,} ({df['RAINFALL_MM'].isna().mean()*100:.2f}%)")

station_meta = df.drop_duplicates("STATION_KEY").set_index("STATION_KEY")[
    ["NAMA_STASIUN", "LATITUDE", "LONGITUDE", "ELEVASI_M"]
]

wide = df.pivot_table(index="TANGGAL", columns="STATION_KEY", values="RAINFALL_MM")
wide = wide.sort_index()


# -------------------------------------------------------------
# 2. INTERPOLASI LINEAR (GAP PENDEK, PER STASIUN)
# -------------------------------------------------------------
log("\n" + "=" * 70)
log(f"STEP 2A - INTERPOLASI LINEAR (gap <= {GAP_INTERPOLASI_MAX} hari)")
log("=" * 70)

n_missing_sebelum = wide.isna().sum().sum()
wide_interp = wide.interpolate(
    method="linear", axis=0, limit=GAP_INTERPOLASI_MAX, limit_area="inside"
)
n_missing_setelah_interp = wide_interp.isna().sum().sum()
n_terisi_interp = n_missing_sebelum - n_missing_setelah_interp

log(f"Missing sebelum interpolasi        : {n_missing_sebelum:,}")
log(f"Terisi oleh interpolasi linear      : {n_terisi_interp:,}")
log(f"Missing tersisa (gap panjang/ujung) : {n_missing_setelah_interp:,}")


# -------------------------------------------------------------
# 3. KNN IMPUTER (SISA GAP PANJANG, MEMANFAATKAN KORELASI ANTAR STASIUN)
# -------------------------------------------------------------
log("\n" + "=" * 70)
log(f"STEP 2B - KNN IMPUTER (k={K_NEIGHBORS_IMPUTE}, korelasi antar stasiun)")
log("=" * 70)

imputer = KNNImputer(n_neighbors=K_NEIGHBORS_IMPUTE, weights="distance")
wide_imputed_arr = imputer.fit_transform(wide_interp)
wide_imputed = pd.DataFrame(wide_imputed_arr, index=wide_interp.index, columns=wide_interp.columns)
wide_imputed = wide_imputed.clip(lower=0)  # curah hujan tidak boleh negatif

n_missing_akhir = wide_imputed.isna().sum().sum()
log(f"Missing value setelah KNN Imputer: {n_missing_akhir:,}")


# -------------------------------------------------------------
# 4. NORMALISASI (MinMaxScaler 0-1, PER STASIUN)
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 2C - NORMALISASI (MinMaxScaler 0-1, per stasiun)")
log("=" * 70)

scaler = MinMaxScaler()
wide_norm_arr = scaler.fit_transform(wide_imputed)
wide_norm = pd.DataFrame(wide_norm_arr, index=wide_imputed.index, columns=wide_imputed.columns)

log("Normalisasi selesai. Rentang nilai tiap stasiun sekarang 0-1.")


# -------------------------------------------------------------
# 5. HITUNG TETANGGA TERDEKAT (HAVERSINE) & FITUR RATA-RATA TETANGGA
# -------------------------------------------------------------
log("\n" + "=" * 70)
log(f"STEP 2D - FITUR TETANGGA TERDEKAT (k={K_NEIGHBORS_FITUR}, jarak Haversine)")
log("=" * 70)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

stations = station_meta.index.tolist()
n_stations = len(stations)
dist_matrix = pd.DataFrame(index=stations, columns=stations, dtype=float)

lat = station_meta["LATITUDE"].values
lon = station_meta["LONGITUDE"].values
for i, s1 in enumerate(stations):
    d = haversine(lat[i], lon[i], lat, lon)
    dist_matrix.loc[s1] = d

neighbor_map = {}
for s in stations:
    d = dist_matrix.loc[s].drop(s).sort_values()
    neighbor_map[s] = d.index[:K_NEIGHBORS_FITUR].tolist()

log(f"Contoh tetangga terdekat untuk 3 stasiun pertama:")
for s in stations[:3]:
    nama = station_meta.loc[s, "NAMA_STASIUN"]
    log(f"  {nama} ({s}) -> tetangga: {[station_meta.loc[n,'NAMA_STASIUN'] for n in neighbor_map[s]]}")

neighbor_avg = pd.DataFrame(index=wide_imputed.index, columns=stations, dtype=float)
neighbor_cols = {}
for s in stations:
    neighbor_cols[s] = wide_imputed[neighbor_map[s]].mean(axis=1)
neighbor_avg = pd.DataFrame(neighbor_cols)

log(f"\nFitur RAINFALL_TETANGGA_AVG berhasil dihitung untuk {n_stations} stasiun.")

neighbor_map_df = pd.DataFrame([
    {"STATION_KEY": s, "TETANGGA_1": neighbor_map[s][0] if len(neighbor_map[s])>0 else None,
     "TETANGGA_2": neighbor_map[s][1] if len(neighbor_map[s])>1 else None,
     "TETANGGA_3": neighbor_map[s][2] if len(neighbor_map[s])>2 else None,
     "TETANGGA_4": neighbor_map[s][3] if len(neighbor_map[s])>3 else None,
     "TETANGGA_5": neighbor_map[s][4] if len(neighbor_map[s])>4 else None}
    for s in stations
])


# -------------------------------------------------------------
# 6. SUSUN DATA FINAL (LONG FORMAT) SIAP UNTUK FEATURE ENGINEERING/MODELING
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 2E - SUSUN DATA FINAL")
log("=" * 70)

df_final = wide_imputed.reset_index().melt(id_vars="TANGGAL", var_name="STATION_KEY", value_name="RAINFALL_MM")
df_norm = wide_norm.reset_index().melt(id_vars="TANGGAL", var_name="STATION_KEY", value_name="RAINFALL_MM_NORM")
df_neighbor = neighbor_avg.reset_index().melt(id_vars="TANGGAL", var_name="STATION_KEY", value_name="RAINFALL_TETANGGA_AVG")
df_flag = wide.isna().reset_index().melt(id_vars="TANGGAL", var_name="STATION_KEY", value_name="IS_IMPUTED")

df_final = (
    df_final
    .merge(df_norm, on=["TANGGAL", "STATION_KEY"])
    .merge(df_neighbor, on=["TANGGAL", "STATION_KEY"])
    .merge(df_flag, on=["TANGGAL", "STATION_KEY"])
    .merge(station_meta.reset_index(), on="STATION_KEY", how="left")
)
df_final = df_final[[
    "STATION_KEY", "NAMA_STASIUN", "LATITUDE", "LONGITUDE", "ELEVASI_M", "TANGGAL",
    "RAINFALL_MM", "RAINFALL_MM_NORM", "RAINFALL_TETANGGA_AVG", "IS_IMPUTED"
]]
df_final = df_final.sort_values(["STATION_KEY", "TANGGAL"]).reset_index(drop=True)

log(f"Shape data final       : {df_final.shape}")
log(f"Total baris hasil imputasi: {df_final['IS_IMPUTED'].sum():,} "
    f"({df_final['IS_IMPUTED'].mean()*100:.2f}%)")
log(f"Missing value tersisa  : {df_final['RAINFALL_MM'].isna().sum():,}")


# -------------------------------------------------------------
# 7. SIMPAN OUTPUT
# -------------------------------------------------------------
path_data_final = os.path.join(OUTPUT_DIR, "02_data_siap_modeling.csv")
path_neighbor_map = os.path.join(OUTPUT_DIR, "02_mapping_tetangga_stasiun.csv")
path_report_txt = os.path.join(REPORT_DIR, "02_laporan_imputasi_normalisasi.txt")

df_final.to_csv(path_data_final, index=False)
neighbor_map_df.to_csv(path_neighbor_map, index=False)

log("\n" + "=" * 70)
log("OUTPUT TERSIMPAN")
log("=" * 70)
log(f"1. Data siap modeling         : {path_data_final}")
log(f"2. Mapping tetangga stasiun   : {path_neighbor_map}")
log(f"3. Laporan proses (txt)       : {path_report_txt}")

with open(path_report_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("\nSELESAI.")