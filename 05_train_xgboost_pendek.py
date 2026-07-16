"""
=============================================================
STEP 5 - TRAINING XGBOOST UNTUK PREDIKSI GENUINE H+1 s/d H+7
Input : 03_Hasil/02_data_siap_modeling.csv
Output: model XGBoost per horizon (tersimpan), laporan akurasi,
        dan prediksi H+1..H+7 terkini (dari data terakhir 2025-12-31)
=============================================================
CARA PAKAI:
1. Taruh file ini satu folder dengan folder 03_Hasil/
2. pip install xgboost joblib scikit-learn
3. Jalankan: python 05_train_xgboost_pendek.py
=============================================================
"""

import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBRegressor

# -------------------------------------------------------------
# 0. KONFIGURASI
# -------------------------------------------------------------
INPUT_PATH = "03_Hasil/02_data_siap_modeling.csv"
OUTPUT_DIR = "03_Hasil"
MODEL_DIR = "03_Hasil/models"
REPORT_DIR = "laporan_hasil"

HORIZON_MAX = 7
HOLDOUT_TAHUN = 2  # konsisten dengan backtest SARIMA sebelumnya

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


def build_features(df):
    """
    Bangun fitur lag, rolling, tetangga spasial (lag), dan kalender.
    Semua fitur dihitung per stasiun secara terpisah (groupby STATION_KEY)
    supaya tidak bocor data antar-stasiun.
    """
    df = df.sort_values(["STATION_KEY", "TANGGAL"]).copy()
    g = df.groupby("STATION_KEY")["RAINFALL_MM"]

    df["LAG_1"] = g.shift(1)
    df["LAG_3"] = g.shift(3)
    df["LAG_7"] = g.shift(7)
    df["ROLL_MEAN_3"] = g.shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    df["ROLL_MEAN_7"] = g.shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    df["ROLL_MEAN_14"] = g.shift(1).rolling(14).mean().reset_index(level=0, drop=True)

    # fitur tetangga: pakai lag 1 hari supaya tidak pakai info "masa depan" tetangga
    if "RAINFALL_TETANGGA_AVG" in df.columns:
        gt = df.groupby("STATION_KEY")["RAINFALL_TETANGGA_AVG"]
        df["TETANGGA_LAG1"] = gt.shift(1)
    else:
        df["TETANGGA_LAG1"] = np.nan

    df["BULAN"] = df["TANGGAL"].dt.month
    doy = df["TANGGAL"].dt.dayofyear
    df["DOY_SIN"] = np.sin(2 * np.pi * doy / 365.25)
    df["DOY_COS"] = np.cos(2 * np.pi * doy / 365.25)
    df["BULAN_SIN"] = np.sin(2 * np.pi * df["BULAN"] / 12)
    df["BULAN_COS"] = np.cos(2 * np.pi * df["BULAN"] / 12)

    return df


FEATURE_COLS = [
    "LAG_1", "LAG_3", "LAG_7", "ROLL_MEAN_3", "ROLL_MEAN_7", "ROLL_MEAN_14",
    "TETANGGA_LAG1", "DOY_SIN", "DOY_COS", "BULAN_SIN", "BULAN_COS"
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    log("=" * 70)
    log("STEP 5 - TRAINING XGBOOST PREDIKSI GENUINE H+1 s/d H+7")
    log("=" * 70)
    log(f"Waktu proses : {pd.Timestamp.now()}")

    df = pd.read_csv(INPUT_PATH, parse_dates=["TANGGAL"])
    station_meta = df.drop_duplicates("STATION_KEY").set_index("STATION_KEY")[
        ["NAMA_STASIUN", "LATITUDE", "LONGITUDE", "ELEVASI_M"]
    ]
    stations = station_meta.index.tolist()
    log(f"Jumlah stasiun : {len(stations)}")
    log(f"Jumlah baris data : {len(df):,}")

    tanggal_terakhir = df["TANGGAL"].max()
    log(f"Tanggal terakhir data historis : {tanggal_terakhir.date()}")

    # -------------------------------------------------------------
    # 1. FEATURE ENGINEERING
    # -------------------------------------------------------------
    log("\n" + "=" * 70)
    log("STEP 5A - FEATURE ENGINEERING (lag, rolling, tetangga, kalender)")
    log("=" * 70)

    df_feat = build_features(df)
    log(f"Fitur dibangun: {FEATURE_COLS}")

    # buat target untuk tiap horizon H+1..H+7
    df_feat = df_feat.sort_values(["STATION_KEY", "TANGGAL"])
    for h in range(1, HORIZON_MAX + 1):
        df_feat[f"TARGET_H{h}"] = df_feat.groupby("STATION_KEY")["RAINFALL_MM"].shift(-h)

    cutoff_holdout = tanggal_terakhir - pd.DateOffset(years=HOLDOUT_TAHUN)
    log(f"Cutoff train/holdout : {cutoff_holdout.date()} "
        f"(train sebelum ini, validasi setelahnya, konsisten dgn backtest SARIMA)")

    # -------------------------------------------------------------
    # 2. TRAINING PER HORIZON (model regional gabungan, bukan per-stasiun)
    # -------------------------------------------------------------
    log("\n" + "=" * 70)
    log("STEP 5B - TRAINING MODEL XGBOOST PER HORIZON (regional gabungan)")
    log("=" * 70)

    metrik_rows = []
    prediksi_terkini_rows = []

    # data untuk prediksi H+1..H+7 dari titik terakhir (per stasiun, baris paling akhir)
    data_terkini = df_feat[df_feat["TANGGAL"] == tanggal_terakhir].copy()

    for h in range(1, HORIZON_MAX + 1):
        target_col = f"TARGET_H{h}"
        subset = df_feat.dropna(subset=FEATURE_COLS + [target_col]).copy()

        train = subset[subset["TANGGAL"] <= cutoff_holdout]
        test = subset[subset["TANGGAL"] > cutoff_holdout]

        if len(train) < 500 or len(test) < 100:
            log(f"H+{h}: data tidak cukup untuk training (train={len(train)}, test={len(test)}), dilewati.")
            continue

        X_train, y_train = train[FEATURE_COLS], train[target_col]
        X_test, y_test = test[FEATURE_COLS], test[target_col]

        model = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            objective="reg:squarederror", n_jobs=-1
        )
        model.fit(X_train, y_train)

        pred_test = np.clip(model.predict(X_test), 0, None)
        mae = np.mean(np.abs(pred_test - y_test))
        rmse = np.sqrt(np.mean((pred_test - y_test) ** 2))
        metrik_rows.append({"HORIZON": f"H+{h}", "MAE_mm": round(mae, 3), "RMSE_mm": round(rmse, 3),
                             "N_TEST": len(test)})
        log(f"H+{h}: MAE={mae:.3f} mm | RMSE={rmse:.3f} mm | n_test={len(test):,}")

        model_path = os.path.join(MODEL_DIR, f"xgb_horizon_h{h}.pkl")
        joblib.dump(model, model_path)

        # prediksi dari data terkini (untuk semua stasiun, horizon ini)
        data_valid = data_terkini.dropna(subset=FEATURE_COLS)
        if len(data_valid) > 0:
            pred_terkini = np.clip(model.predict(data_valid[FEATURE_COLS]), 0, None)
            for stat_key, pred_val in zip(data_valid["STATION_KEY"], pred_terkini):
                prediksi_terkini_rows.append({
                    "STATION_KEY": stat_key,
                    "HORIZON": h,
                    "TANGGAL_PREDIKSI": tanggal_terakhir + pd.Timedelta(days=h),
                    "PREDIKSI_XGBOOST_MM": pred_val
                })

    metrik_df = pd.DataFrame(metrik_rows)
    prediksi_df = pd.DataFrame(prediksi_terkini_rows)
    prediksi_df = prediksi_df.merge(station_meta.reset_index(), on="STATION_KEY", how="left")

    log(f"\nRingkasan akurasi semua horizon:")
    if len(metrik_df) > 0:
        log(f"  MAE rata-rata seluruh horizon  : {metrik_df['MAE_mm'].mean():.3f} mm")
        log(f"  RMSE rata-rata seluruh horizon : {metrik_df['RMSE_mm'].mean():.3f} mm")

    # -------------------------------------------------------------
    # 3. SIMPAN OUTPUT
    # -------------------------------------------------------------
    path_metrik = os.path.join(OUTPUT_DIR, "05_akurasi_xgboost_per_horizon.csv")
    path_prediksi = os.path.join(OUTPUT_DIR, "05_prediksi_jangka_pendek_terkini.csv")
    path_report = os.path.join(REPORT_DIR, "05_laporan_training_xgboost.txt")

    metrik_df.to_csv(path_metrik, index=False)
    prediksi_df.to_csv(path_prediksi, index=False)

    log("\n" + "=" * 70)
    log("OUTPUT TERSIMPAN")
    log("=" * 70)
    log(f"1. Model XGBoost per horizon (7 file) : {MODEL_DIR}/xgb_horizon_h1.pkl s/d h7.pkl")
    log(f"2. Laporan akurasi per horizon         : {path_metrik}")
    log(f"3. Prediksi H+1..H+7 terkini            : {path_prediksi}")
    log(f"4. Laporan proses (txt)                 : {path_report}")

    with open(path_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\nSELESAI.")


if __name__ == "__main__":
    main()
