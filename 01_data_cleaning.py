"""
=============================================================
STEP 1 - DATA CLEANING: DATA CURAH HUJAN SUMATERA BARAT
Sumber: gabungan_11_file.csv (2015-2025, 164 POS HUJAN ID)
Output: data bersih (long format harian) + laporan kelengkapan data
=============================================================
"""

import pandas as pd
import numpy as np
import os

# -------------------------------------------------------------
# 0. KONFIGURASI
# -------------------------------------------------------------
INPUT_PATH = "gabungan_11_file.csv"   # <-- sesuaikan lokasi file di komputermu
OUTPUT_DIR = "03_Hasil"
REPORT_DIR = "laporan_hasil"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TANGGAL_MULAI = "2015-01-01"
TANGGAL_AKHIR = "2025-12-31"
THRESHOLD_KELENGKAPAN = 0.50  # 50% data lengkap -> kriteria seleksi pos hujan terpakai

report_lines = []
def log(msg=""):
    """Cetak ke layar sekaligus simpan ke laporan txt."""
    print(msg)
    report_lines.append(str(msg))


# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
log("=" * 70)
log("STEP 1 - DATA CLEANING: CURAH HUJAN SUMATERA BARAT")
log("=" * 70)
log(f"Waktu proses    : {pd.Timestamp.now()}")
log(f"File input      : {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)
df["DATA TIMESTAMP"] = pd.to_datetime(
    df["DATA TIMESTAMP"], errors="coerce", utc=True
).dt.tz_localize(None)
df["DATA TIMESTAMP"] = df["DATA TIMESTAMP"].dt.normalize()

log(f"\nJumlah baris awal        : {len(df):,}")
log(f"Jumlah POS HUJAN ID awal : {df['POS HUJAN ID'].nunique()}")
log(f"Jumlah NAME unik awal    : {df['NAME'].nunique()}")


# -------------------------------------------------------------
# 2. GABUNGKAN STASIUN DENGAN NAMA SAMA TAPI ID BERBEDA
#    (kemungkinan pergantian alat / relokasi sensor)
#    Aturan: gabung timeline; jika tanggal sama-sama ada di 2 ID
#    (overlap), prioritaskan ID dengan total data historis lebih panjang
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 2 - PENGGABUNGAN STASIUN DUPLIKAT (NAME sama, ID beda)")
log("=" * 70)

name_to_ids = df.groupby("NAME")["POS HUJAN ID"].unique()
dup_names = name_to_ids[name_to_ids.apply(len) > 1]

log(f"Ditemukan {len(dup_names)} nama stasiun dengan >1 ID:")

id_rank = df.groupby("POS HUJAN ID")["DATA TIMESTAMP"].count()

canonical_id_map = {}
for name, ids in dup_names.items():
    ids_sorted = sorted(ids, key=lambda x: id_rank.get(x, 0), reverse=True)
    id_utama = ids_sorted[0]
    for i in ids:
        canonical_id_map[i] = id_utama
    log(f"  - {name}: {list(ids)} -> digabung, ID utama = {id_utama}")

df["STATION_KEY"] = df["POS HUJAN ID"].map(lambda x: canonical_id_map.get(x, x))

df["_prioritas"] = df.apply(
    lambda r: 1 if r["POS HUJAN ID"] == canonical_id_map.get(r["POS HUJAN ID"], r["POS HUJAN ID"]) else 0,
    axis=1
)
df = df.sort_values(["STATION_KEY", "DATA TIMESTAMP", "_prioritas"], ascending=[True, True, False])
before_dedup = len(df)
df = df.drop_duplicates(subset=["STATION_KEY", "DATA TIMESTAMP"], keep="first")
log(f"\nBaris sebelum gabung overlap : {before_dedup:,}")
log(f"Baris setelah gabung overlap : {len(df):,} (selisih {before_dedup - len(df):,} baris overlap terbuang)")

meta_cols = ["NAME", "CURRENT LATITUDE", "CURRENT LONGITUDE", "CURRENT ELEVATION M"]
station_meta = (
    df[df["POS HUJAN ID"] == df["STATION_KEY"]]
    .drop_duplicates(subset="STATION_KEY")[["STATION_KEY"] + meta_cols]
    .set_index("STATION_KEY")
)

log(f"\nJumlah stasiun setelah penggabungan: {df['STATION_KEY'].nunique()}")


# -------------------------------------------------------------
# 3. TANGANI SENTINEL VALUE
#    8888 = di bawah ambang ukur alat -> dianggap TIDAK HUJAN (0 mm)
#    9999 = data tidak terukur/rusak -> dianggap MISSING (NaN), diimputasi di Step 2
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 3 - PENANGANAN SENTINEL VALUE (8888 / 9999)")
log("=" * 70)

sentinel_nol = [8888, -8888]
sentinel_nan = [9999, -9999]

n_nol = df["RAINFALL DAY MM"].isin(sentinel_nol).sum()
n_nan = df["RAINFALL DAY MM"].isin(sentinel_nan).sum()
log(f"  Nilai 8888 (di bawah ambang ukur): {n_nol:,} baris -> di-set 0 mm")
log(f"  Nilai 9999 (data tidak terukur)  : {n_nan:,} baris -> di-set NaN (diimputasi di Step 2)")

df.loc[df["RAINFALL DAY MM"].isin(sentinel_nol), "RAINFALL DAY MM"] = 0.0
df.loc[df["RAINFALL DAY MM"].isin(sentinel_nan), "RAINFALL DAY MM"] = np.nan
log(f"\nTotal baris sentinel: {n_nol + n_nan:,}")


# -------------------------------------------------------------
# 4. TANGANI RAINFALL TRACE = 'Y' -> curah hujan diset 0 mm
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 4 - PENANGANAN RAINFALL TRACE = 'Y'")
log("=" * 70)

n_trace = (df["RAINFALL TRACE"] == "Y").sum()
log(f"Baris dengan TRACE = 'Y' : {n_trace:,} -> RAINFALL DAY MM di-set 0 mm")
df.loc[df["RAINFALL TRACE"] == "Y", "RAINFALL DAY MM"] = 0.0


# -------------------------------------------------------------
# 5. BUAT DERET HARIAN LENGKAP PER STASIUN (2015-01-01 s/d 2025-12-31)
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 5 - REINDEX KE DERET HARIAN PENUH & HITUNG KELENGKAPAN")
log("=" * 70)

full_dates = pd.date_range(TANGGAL_MULAI, TANGGAL_AKHIR, freq="D")
total_hari = len(full_dates)
log(f"Total hari acuan ({TANGGAL_MULAI} s/d {TANGGAL_AKHIR}): {total_hari:,} hari")

df_pivot = df.pivot_table(
    index="DATA TIMESTAMP", columns="STATION_KEY", values="RAINFALL DAY MM", aggfunc="first"
)
df_pivot = df_pivot.reindex(full_dates)
df_pivot.index.name = "TANGGAL"

completeness = df_pivot.notna().sum() / total_hari
completeness_report = (
    completeness.sort_values(ascending=False)
    .rename("PERSEN_LENGKAP")
    .reset_index()
)
completeness_report["PERSEN_LENGKAP"] = (completeness_report["PERSEN_LENGKAP"] * 100).round(2)
completeness_report = completeness_report.merge(
    station_meta.reset_index(), on="STATION_KEY", how="left"
)

n_stasiun_kosong = df["STATION_KEY"].nunique() - df_pivot.shape[1]
log(f"\nStasiun dengan data 100% kosong/sentinel (dibuang otomatis): {n_stasiun_kosong}")
log(f"Statistik kelengkapan data seluruh {df_pivot.shape[1]} stasiun (dengan data valid):")
log(completeness_report["PERSEN_LENGKAP"].describe().round(2).to_string())


# -------------------------------------------------------------
# 6. SELEKSI STASIUN FINAL (>= threshold kelengkapan data)
# -------------------------------------------------------------
log("\n" + "=" * 70)
log(f"STEP 6 - SELEKSI STASIUN FINAL (threshold >= {THRESHOLD_KELENGKAPAN*100:.0f}% lengkap)")
log("=" * 70)

stasiun_terpilih = completeness_report[
    completeness_report["PERSEN_LENGKAP"] >= THRESHOLD_KELENGKAPAN * 100
]["STATION_KEY"].tolist()

log(f"Jumlah stasiun lolos threshold : {len(stasiun_terpilih)}")
log(f"Jumlah stasiun TIDAK lolos     : {df_pivot.shape[1] - len(stasiun_terpilih)}")


# -------------------------------------------------------------
# 7. SUSUN DATA BERSIH FINAL (long format) UNTUK STASIUN TERPILIH
# -------------------------------------------------------------
df_final = df_pivot[stasiun_terpilih].reset_index().melt(
    id_vars="TANGGAL", var_name="STATION_KEY", value_name="RAINFALL_MM"
)
df_final = df_final.merge(station_meta.reset_index(), on="STATION_KEY", how="left")
df_final = df_final.rename(columns={
    "NAME": "NAMA_STASIUN",
    "CURRENT LATITUDE": "LATITUDE",
    "CURRENT LONGITUDE": "LONGITUDE",
    "CURRENT ELEVATION M": "ELEVASI_M",
})
df_final = df_final[["STATION_KEY", "NAMA_STASIUN", "LATITUDE", "LONGITUDE", "ELEVASI_M", "TANGGAL", "RAINFALL_MM"]]
df_final = df_final.sort_values(["STATION_KEY", "TANGGAL"]).reset_index(drop=True)

log(f"\nShape data bersih final : {df_final.shape}")
log(f"Jumlah stasiun final    : {df_final['STATION_KEY'].nunique()}")
log(f"Missing value tersisa (NaN, murni krn tak ada laporan): {df_final['RAINFALL_MM'].isna().sum():,} "
    f"({df_final['RAINFALL_MM'].isna().mean()*100:.2f}%)")


# -------------------------------------------------------------
# 8. SIMPAN OUTPUT
# -------------------------------------------------------------
path_data_bersih = os.path.join(OUTPUT_DIR, "01_data_bersih_curah_hujan_sumbar.csv")
path_completeness = os.path.join(OUTPUT_DIR, "01_kelengkapan_data_semua_stasiun.csv")
path_report_txt = os.path.join(REPORT_DIR, "01_laporan_data_cleaning.txt")

df_final.to_csv(path_data_bersih, index=False)
completeness_report.sort_values("PERSEN_LENGKAP", ascending=False).to_csv(path_completeness, index=False)

log("\n" + "=" * 70)
log("STEP 7 - OUTPUT TERSIMPAN")
log("=" * 70)
log(f"1. Data bersih (long format)         : {path_data_bersih}")
log(f"2. Laporan kelengkapan semua stasiun : {path_completeness}")
log(f"3. Laporan proses (txt)              : {path_report_txt}")

with open(path_report_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("\nSELESAI.")