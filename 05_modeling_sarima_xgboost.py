"""
=============================================================
STEP 5 - MODELING HYBRID SARIMA-XGBOOST
Input : 03_Hasil/04_data_siap_modeling_final.csv (dari Step 4)
Output: model terlatih (.pkl), laporan evaluasi, feature importance
=============================================================
Arsitektur:
  1. SARIMA -> komponen tren/musiman REGIONAL (rata-rata seluruh stasiun,
     agregasi bulanan) -> fitur SARIMA_MUSIMAN & SARIMA_RESIDU per baris harian
  2. XGBoost -> klasifikasi biner LABEL_EKSTREM_100_7HARI (hujan >=100mm
     dalam 7 hari ke depan), memakai fitur lag/rolling/spasial + fitur SARIMA
  3. Validasi: TimeSeriesSplit berbasis tanggal (bukan random split)
=============================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
import pickle
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve, f1_score
)
import xgboost as xgb

# -------------------------------------------------------------
# 0. KONFIGURASI
# -------------------------------------------------------------
INPUT_PATH = "03_Hasil/04_data_siap_modeling_final.csv"
OUTPUT_DIR = "03_Hasil"
MODEL_DIR = "03_Hasil/model"
REPORT_DIR = "laporan_hasil"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TARGET_COL = "LABEL_EKSTREM_100_7HARI"
N_SPLITS_CV = 5
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)  # musiman tahunan, di data BULANAN (m=12)

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
log("=" * 70)
log("STEP 5 - MODELING HYBRID SARIMA-XGBOOST")
log("=" * 70)
log(f"Waktu proses : {pd.Timestamp.now()}")
log(f"File input   : {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH, parse_dates=["TANGGAL"])
log(f"\nShape data awal : {df.shape}")
log(f"Jumlah stasiun   : {df['STATION_KEY'].nunique()}")
log(f"Target           : {TARGET_COL}")


# -------------------------------------------------------------
# 2. KOMPONEN SARIMA (REGIONAL, AGREGASI BULANAN)
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 5A - SARIMA: TREN & MUSIMAN REGIONAL (agregasi bulanan)")
log("=" * 70)

regional_daily = df.groupby("TANGGAL")["RAINFALL_MM"].mean()
regional_monthly = regional_daily.resample("MS").mean()

log(f"Jumlah titik data bulanan untuk SARIMA: {len(regional_monthly)}")
log(f"Order SARIMA         : {SARIMA_ORDER}")
log(f"Seasonal order SARIMA: {SARIMA_SEASONAL_ORDER} (musiman tahunan, m=12)")

sarima_model = SARIMAX(
    regional_monthly, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER,
    enforce_stationarity=False, enforce_invertibility=False
)
sarima_fit = sarima_model.fit(disp=False)
log("\nSARIMA berhasil dilatih pada rata-rata curah hujan regional bulanan.")
log(sarima_fit.summary().tables[0].as_text())

sarima_fitted_monthly = sarima_fit.fittedvalues
sarima_fitted_monthly.name = "SARIMA_MUSIMAN"

log(f"\n[CATATAN METODOLOGI] SARIMA dilatih pada SELURUH periode 2015-2025 sekaligus")
log("(bukan rolling/incremental) karena fungsinya di sini adalah dekomposisi")
log("tren-musiman regional sebagai FITUR untuk XGBoost, bukan sebagai model")
log("forecast independen. Untuk deployment produksi, SARIMA sebaiknya di-refit")
log("berkala (mis. tiap awal bulan) dengan data terbaru.")

sarima_daily = sarima_fitted_monthly.reindex(
    pd.date_range(regional_monthly.index.min(), df["TANGGAL"].max(), freq="D"),
    method="ffill"
)
sarima_daily.index.name = "TANGGAL"
sarima_daily_df = sarima_daily.reset_index()

df = df.merge(sarima_daily_df, on="TANGGAL", how="left")
df["SARIMA_MUSIMAN"] = df["SARIMA_MUSIMAN"].ffill().bfill()
df["SARIMA_RESIDU"] = df["RAINFALL_MM"] - df["SARIMA_MUSIMAN"]

log(f"\nFitur SARIMA berhasil ditambahkan ke data harian:")
log(f"  - SARIMA_MUSIMAN : nilai tren-musiman regional (mm/hari, hasil SARIMA bulanan)")
log(f"  - SARIMA_RESIDU  : RAINFALL_MM - SARIMA_MUSIMAN (deviasi lokal dari baseline regional)")

with open(os.path.join(MODEL_DIR, "sarima_model.pkl"), "wb") as f:
    pickle.dump(sarima_fit, f)
log(f"\nModel SARIMA disimpan: {MODEL_DIR}/sarima_model.pkl")


# -------------------------------------------------------------
# 3. SIAPKAN DATA UNTUK XGBOOST (BUANG BARIS LABEL KOSONG)
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 5B - PERSIAPAN DATA XGBOOST")
log("=" * 70)

n_sebelum = len(df)
df_train_ready = df.dropna(subset=[TARGET_COL]).copy()
df_train_ready[TARGET_COL] = df_train_ready[TARGET_COL].astype(int)
log(f"Baris sebelum buang (label kosong, 7 hari terakhir tiap stasiun): {n_sebelum:,}")
log(f"Baris setelah buang                                             : {len(df_train_ready):,}")

fitur_dikecualikan = [
    "STATION_KEY", "NAMA_STASIUN", "LATITUDE", "LONGITUDE", "ELEVASI_M", "TANGGAL",
    "RAINFALL_MM", "RAINFALL_MM_NORM", "IS_IMPUTED",
    "TARGET_H1", "TARGET_H2", "TARGET_H3", "TARGET_H4", "TARGET_H5", "TARGET_H6", "TARGET_H7",
    "MAX_RAINFALL_7HARI_KEDEPAN", "LABEL_EKSTREM_100_7HARI", "LABEL_EKSTREM_150_7HARI"
]
fitur_cols = [c for c in df_train_ready.columns if c not in fitur_dikecualikan]
log(f"\nJumlah fitur dipakai XGBoost: {len(fitur_cols)}")
log(f"Daftar fitur: {fitur_cols}")

df_train_ready = df_train_ready.sort_values("TANGGAL").reset_index(drop=True)


# -------------------------------------------------------------
# 4. TIME SERIES CROSS VALIDATION (BERBASIS TANGGAL, BUKAN RANDOM)
# -------------------------------------------------------------
log("\n" + "=" * 70)
log(f"STEP 5C - TIME SERIES CROSS VALIDATION ({N_SPLITS_CV} fold)")
log("=" * 70)

tanggal_unik = np.sort(df_train_ready["TANGGAL"].unique())
tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)

hasil_cv = []
fold_ke = 0
for train_idx_date, test_idx_date in tscv.split(tanggal_unik):
    fold_ke += 1
    tgl_train = tanggal_unik[train_idx_date]
    tgl_test = tanggal_unik[test_idx_date]

    train_mask = df_train_ready["TANGGAL"].isin(tgl_train)
    test_mask = df_train_ready["TANGGAL"].isin(tgl_test)

    X_train, y_train = df_train_ready.loc[train_mask, fitur_cols], df_train_ready.loc[train_mask, TARGET_COL]
    X_test, y_test = df_train_ready.loc[test_mask, fitur_cols], df_train_ready.loc[test_mask, TARGET_COL]

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    log(f"\n--- Fold {fold_ke} ---")
    log(f"  Periode train: {tgl_train.min()} s/d {tgl_train.max()} ({len(X_train):,} baris, {n_pos:,} positif)")
    log(f"  Periode test : {tgl_test.min()} s/d {tgl_test.max()} ({len(X_test):,} baris, {y_test.sum():,} positif)")
    log(f"  ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | F1(thr=0.5): {f1:.4f}")

    hasil_cv.append({"fold": fold_ke, "roc_auc": roc_auc, "pr_auc": pr_auc, "f1": f1,
                      "n_train": len(X_train), "n_test": len(X_test)})

cv_df = pd.DataFrame(hasil_cv)
log("\n" + "=" * 70)
log("RINGKASAN CROSS VALIDATION (rata-rata seluruh fold)")
log("=" * 70)
log(f"ROC-AUC rata-rata: {cv_df['roc_auc'].mean():.4f} (+/- {cv_df['roc_auc'].std():.4f})")
log(f"PR-AUC rata-rata : {cv_df['pr_auc'].mean():.4f} (+/- {cv_df['pr_auc'].std():.4f})")
log(f"F1 rata-rata      : {cv_df['f1'].mean():.4f} (+/- {cv_df['f1'].std():.4f})")

cv_df.to_csv(os.path.join(OUTPUT_DIR, "05_hasil_cross_validation.csv"), index=False)


# -------------------------------------------------------------
# 5. MODEL FINAL (DILATIH PADA SELURUH DATA) + THRESHOLD TUNING
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 5D - MODEL FINAL (dilatih pada seluruh data) & THRESHOLD TUNING")
log("=" * 70)

split_list = list(tscv.split(tanggal_unik))
tgl_train_final = tanggal_unik[split_list[-1][0]]
tgl_val_final = tanggal_unik[split_list[-1][1]]

train_mask = df_train_ready["TANGGAL"].isin(tgl_train_final)
val_mask = df_train_ready["TANGGAL"].isin(tgl_val_final)

X_train_final = df_train_ready.loc[train_mask, fitur_cols]
y_train_final = df_train_ready.loc[train_mask, TARGET_COL]
X_val_final = df_train_ready.loc[val_mask, fitur_cols]
y_val_final = df_train_ready.loc[val_mask, TARGET_COL]

scale_pos_weight_final = (len(y_train_final) - y_train_final.sum()) / max(y_train_final.sum(), 1)

model_final = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight_final,
    eval_metric="aucpr", random_state=42, n_jobs=-1
)
model_final.fit(X_train_final, y_train_final)

y_val_prob = model_final.predict_proba(X_val_final)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val_final, y_val_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
idx_terbaik = np.argmax(f1_scores[:-1])
threshold_terbaik = thresholds[idx_terbaik]

log(f"Threshold terbaik (memaksimalkan F1 di data validasi): {threshold_terbaik:.3f}")
log(f"  Pada threshold ini -> Precision: {precisions[idx_terbaik]:.4f}, Recall: {recalls[idx_terbaik]:.4f}, "
    f"F1: {f1_scores[idx_terbaik]:.4f}")

y_val_pred_default = (y_val_prob >= 0.5).astype(int)
y_val_pred_tuned = (y_val_prob >= threshold_terbaik).astype(int)

log(f"\n--- Classification report (threshold default 0.5) ---")
log(classification_report(y_val_final, y_val_pred_default, digits=4))
log(f"\n--- Classification report (threshold tuned {threshold_terbaik:.3f}) ---")
log(classification_report(y_val_final, y_val_pred_tuned, digits=4))

cm = confusion_matrix(y_val_final, y_val_pred_tuned)
log(f"\nConfusion Matrix (threshold tuned):")
log(f"                 Prediksi Tidak Ekstrem | Prediksi Ekstrem")
log(f"Aktual Tidak Ekstrem:  {cm[0,0]:>10,}         | {cm[0,1]:>10,}")
log(f"Aktual Ekstrem:        {cm[1,0]:>10,}         | {cm[1,1]:>10,}")

log(f"\n[CATATAN MITIGASI BENCANA] Untuk konteks early warning, RECALL lebih diprioritaskan")
log("daripada precision (lebih baik false alarm daripada lolos kejadian ekstrem).")
log("Threshold bisa diturunkan dari nilai optimal-F1 di atas jika BPBD/BMKG ingin")
log("recall lebih tinggi meski precision turun -- ini keputusan operasional, bukan teknis.")


# -------------------------------------------------------------
# 6. FEATURE IMPORTANCE
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 5E - FEATURE IMPORTANCE (XGBoost)")
log("=" * 70)

importance = pd.DataFrame({
    "fitur": fitur_cols,
    "importance": model_final.feature_importances_
}).sort_values("importance", ascending=False)

log(importance.head(15).to_string(index=False))
importance.to_csv(os.path.join(OUTPUT_DIR, "05_feature_importance.csv"), index=False)


# -------------------------------------------------------------
# 7. SIMPAN MODEL FINAL & METADATA
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 5F - SIMPAN MODEL & METADATA")
log("=" * 70)

with open(os.path.join(MODEL_DIR, "xgboost_model_final.pkl"), "wb") as f:
    pickle.dump(model_final, f)

metadata = {
    "fitur_cols": fitur_cols,
    "target_col": TARGET_COL,
    "threshold_terbaik": float(threshold_terbaik),
    "batas_ekstrem_utama_mm": 100,
    "batas_ekstrem_info_mm": 150,
    "horizon_hari": 7,
    "sarima_order": SARIMA_ORDER,
    "sarima_seasonal_order": SARIMA_SEASONAL_ORDER,
    "tanggal_training": str(pd.Timestamp.now()),
}
with open(os.path.join(MODEL_DIR, "metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)

log(f"1. Model XGBoost final : {MODEL_DIR}/xgboost_model_final.pkl")
log(f"2. Model SARIMA        : {MODEL_DIR}/sarima_model.pkl")
log(f"3. Metadata model      : {MODEL_DIR}/metadata.pkl")

df.to_csv(os.path.join(OUTPUT_DIR, "05_data_dengan_fitur_sarima.csv"), index=False)
log(f"4. Dataset + fitur SARIMA: {OUTPUT_DIR}/05_data_dengan_fitur_sarima.csv")


# -------------------------------------------------------------
# 8. SIMPAN LAPORAN
# -------------------------------------------------------------
path_report_txt = os.path.join(REPORT_DIR, "05_laporan_modeling_sarima_xgboost.txt")
with open(path_report_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

log(f"\n5. Laporan proses (txt): {path_report_txt}")
print("\nSELESAI.")