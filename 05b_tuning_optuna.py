"""
=============================================================
STEP 5B - HYPERPARAMETER TUNING XGBOOST (OPTUNA)
Input : 03_Hasil/05_data_dengan_fitur_sarima.csv (dari Step 5, sudah ada fitur SARIMA)
Output: model XGBoost final hasil tuning + laporan tuning
=============================================================
Objective  : maksimalkan rata-rata PR-AUC (average precision) di fold
             TimeSeriesSplit terakhir (paling relevan dgn kondisi terkini)
Search space: n_estimators, max_depth, learning_rate, subsample,
              colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda
=============================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
import pickle
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve, f1_score
)
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -------------------------------------------------------------
# 0. KONFIGURASI
# -------------------------------------------------------------
INPUT_PATH = "03_Hasil/05_data_dengan_fitur_sarima.csv"
OUTPUT_DIR = "03_Hasil"
MODEL_DIR = "03_Hasil/model"
REPORT_DIR = "laporan_hasil"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TARGET_COL = "LABEL_EKSTREM_100_7HARI"
N_SPLITS_TUNING = 2     # dipakai selama proses tuning (hemat waktu)
N_TRIALS = 12
TIMEOUT_DETIK = 480     # batas waktu tuning (detik)
FRAC_SAMPLING_TUNING = 0.35   # sampling baris negatif saat search hyperparameter (percepat)

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


# -------------------------------------------------------------
# 1. LOAD DATA (SUDAH ADA FITUR SARIMA DARI STEP 5)
# -------------------------------------------------------------
log("=" * 70)
log("STEP 5B - HYPERPARAMETER TUNING XGBOOST (OPTUNA)")
log("=" * 70)
log(f"Waktu proses : {pd.Timestamp.now()}")
log(f"File input   : {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH, parse_dates=["TANGGAL"])
df_ready = df.dropna(subset=[TARGET_COL]).copy()
df_ready[TARGET_COL] = df_ready[TARGET_COL].astype(int)
df_ready = df_ready.sort_values("TANGGAL").reset_index(drop=True)

fitur_dikecualikan = [
    "STATION_KEY", "NAMA_STASIUN", "LATITUDE", "LONGITUDE", "ELEVASI_M", "TANGGAL",
    "RAINFALL_MM", "RAINFALL_MM_NORM", "IS_IMPUTED",
    "TARGET_H1", "TARGET_H2", "TARGET_H3", "TARGET_H4", "TARGET_H5", "TARGET_H6", "TARGET_H7",
    "MAX_RAINFALL_7HARI_KEDEPAN", "LABEL_EKSTREM_100_7HARI", "LABEL_EKSTREM_150_7HARI"
]
fitur_cols = [c for c in df_ready.columns if c not in fitur_dikecualikan]
log(f"Jumlah baris  : {len(df_ready):,}")
log(f"Jumlah fitur  : {len(fitur_cols)}")

tanggal_unik = np.sort(df_ready["TANGGAL"].unique())
tscv = TimeSeriesSplit(n_splits=N_SPLITS_TUNING)
split_list = list(tscv.split(tanggal_unik))

fold_data = []
for train_idx, test_idx in split_list:
    tgl_train = tanggal_unik[train_idx]
    tgl_test = tanggal_unik[test_idx]
    train_mask = df_ready["TANGGAL"].isin(tgl_train)
    test_mask = df_ready["TANGGAL"].isin(tgl_test)
    X_train = df_ready.loc[train_mask, fitur_cols]
    y_train = df_ready.loc[train_mask, TARGET_COL]
    X_test = df_ready.loc[test_mask, fitur_cols]
    y_test = df_ready.loc[test_mask, TARGET_COL]
    fold_data.append((X_train, y_train, X_test, y_test))

log(f"\nJumlah fold untuk tuning: {N_SPLITS_TUNING}")
for i, (Xtr, ytr, Xte, yte) in enumerate(fold_data, 1):
    log(f"  Fold {i}: train={len(Xtr):,} ({ytr.sum():,} positif) | test={len(Xte):,} ({yte.sum():,} positif)")

# semua baris positif (kejadian ekstrem) dipertahankan, hanya baris negatif yang disampling
fold_data_sampled = []
rng = np.random.RandomState(42)
for X_train, y_train, X_test, y_test in fold_data:
    pos_idx = y_train[y_train == 1].index
    neg_idx = y_train[y_train == 0].index
    neg_sampled = rng.choice(neg_idx, size=int(len(neg_idx) * FRAC_SAMPLING_TUNING), replace=False)
    keep_idx = np.concatenate([pos_idx, neg_sampled])
    fold_data_sampled.append((X_train.loc[keep_idx], y_train.loc[keep_idx], X_test, y_test))

log(f"\nSampling data training saat SEARCH (frac negatif={FRAC_SAMPLING_TUNING}, semua positif dipertahankan):")
for i, (Xtr, ytr, Xte, yte) in enumerate(fold_data_sampled, 1):
    log(f"  Fold {i} (sampled): train={len(Xtr):,} ({ytr.sum():,} positif)")


# -------------------------------------------------------------
# 2. OBJECTIVE FUNCTION OPTUNA
# -------------------------------------------------------------
log("\n" + "=" * 70)
log(f"STEP 5B-1 - OPTUNA SEARCH ({N_TRIALS} trial, target maksimal PR-AUC)")
log("=" * 70)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 350),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    pr_auc_scores = []
    for X_train, y_train, X_test, y_test in fold_data_sampled:
        scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
        model = xgb.XGBClassifier(
            **params, scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr", random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        pr_auc_scores.append(average_precision_score(y_test, y_prob))

    return float(np.mean(pr_auc_scores))

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_DETIK, show_progress_bar=False)

log(f"\nJumlah trial selesai: {len(study.trials)}")
log(f"PR-AUC terbaik (rata-rata {N_SPLITS_TUNING} fold): {study.best_value:.4f}")
log(f"\nHyperparameter terbaik:")
for k, v in study.best_params.items():
    log(f"  {k}: {v}")

trials_df = study.trials_dataframe().sort_values("value", ascending=False)
trials_df.to_csv(os.path.join(OUTPUT_DIR, "05b_optuna_trials.csv"), index=False)


# -------------------------------------------------------------
# 3. LATIH MODEL FINAL DENGAN HYPERPARAMETER TERBAIK
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 5B-2 - MODEL FINAL (hyperparameter hasil tuning)")
log("=" * 70)

X_train_final, y_train_final, X_val_final, y_val_final = fold_data[-1]
scale_pos_weight_final = (len(y_train_final) - y_train_final.sum()) / max(y_train_final.sum(), 1)

model_final = xgb.XGBClassifier(
    **study.best_params, scale_pos_weight=scale_pos_weight_final,
    eval_metric="aucpr", random_state=42, n_jobs=-1
)
model_final.fit(X_train_final, y_train_final)

y_val_prob = model_final.predict_proba(X_val_final)[:, 1]
roc_auc = roc_auc_score(y_val_final, y_val_prob)
pr_auc = average_precision_score(y_val_final, y_val_prob)

precisions, recalls, thresholds = precision_recall_curve(y_val_final, y_val_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
idx_terbaik = np.argmax(f1_scores[:-1])
threshold_terbaik = thresholds[idx_terbaik]

log(f"ROC-AUC (fold terakhir): {roc_auc:.4f}")
log(f"PR-AUC  (fold terakhir): {pr_auc:.4f}")
log(f"Threshold terbaik (F1) : {threshold_terbaik:.3f}")
log(f"  Precision: {precisions[idx_terbaik]:.4f} | Recall: {recalls[idx_terbaik]:.4f} | F1: {f1_scores[idx_terbaik]:.4f}")

y_val_pred_tuned = (y_val_prob >= threshold_terbaik).astype(int)
log(f"\n--- Classification report (threshold tuned {threshold_terbaik:.3f}) ---")
log(classification_report(y_val_final, y_val_pred_tuned, digits=4))

cm = confusion_matrix(y_val_final, y_val_pred_tuned)
log(f"\nConfusion Matrix:")
log(f"                 Prediksi Tidak Ekstrem | Prediksi Ekstrem")
log(f"Aktual Tidak Ekstrem:  {cm[0,0]:>10,}         | {cm[0,1]:>10,}")
log(f"Aktual Ekstrem:        {cm[1,0]:>10,}         | {cm[1,1]:>10,}")

# -------------------------------------------------------------
# 3b. VALIDASI TAMBAHAN: 5-FOLD PENUH DENGAN HYPERPARAMETER TERBAIK
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 5B-3 - VALIDASI 5-FOLD PENUH (hyperparameter hasil tuning)")
log("=" * 70)

tscv5 = TimeSeriesSplit(n_splits=5)
hasil_cv_tuned = []
for i, (train_idx, test_idx) in enumerate(tscv5.split(tanggal_unik), 1):
    tgl_train = tanggal_unik[train_idx]
    tgl_test = tanggal_unik[test_idx]
    train_mask = df_ready["TANGGAL"].isin(tgl_train)
    test_mask = df_ready["TANGGAL"].isin(tgl_test)
    X_train = df_ready.loc[train_mask, fitur_cols]
    y_train = df_ready.loc[train_mask, TARGET_COL]
    X_test = df_ready.loc[test_mask, fitur_cols]
    y_test = df_ready.loc[test_mask, TARGET_COL]

    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    model = xgb.XGBClassifier(
        **study.best_params, scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    log(f"  Fold {i}: ROC-AUC={roc:.4f} | PR-AUC={pr:.4f} | F1(0.5)={f1:.4f}")
    hasil_cv_tuned.append({"fold": i, "roc_auc": roc, "pr_auc": pr, "f1": f1})

cv_tuned_df = pd.DataFrame(hasil_cv_tuned)
log(f"\nRata-rata 5-fold (tuned): ROC-AUC={cv_tuned_df['roc_auc'].mean():.4f}, "
    f"PR-AUC={cv_tuned_df['pr_auc'].mean():.4f}, F1={cv_tuned_df['f1'].mean():.4f}")
cv_tuned_df.to_csv(os.path.join(OUTPUT_DIR, "05b_cv_hasil_tuned.csv"), index=False)


# -------------------------------------------------------------
# 4. FEATURE IMPORTANCE (MODEL TUNED)
# -------------------------------------------------------------
importance = pd.DataFrame({
    "fitur": fitur_cols, "importance": model_final.feature_importances_
}).sort_values("importance", ascending=False)
log(f"\nTop 15 fitur (model tuned):")
log(importance.head(15).to_string(index=False))
importance.to_csv(os.path.join(OUTPUT_DIR, "05b_feature_importance_tuned.csv"), index=False)


# -------------------------------------------------------------
# 5. SIMPAN MODEL & METADATA (OVERWRITE JADI MODEL FINAL RESMI)
# -------------------------------------------------------------
with open(os.path.join(MODEL_DIR, "xgboost_model_final.pkl"), "wb") as f:
    pickle.dump(model_final, f)

metadata = {
    "fitur_cols": fitur_cols,
    "target_col": TARGET_COL,
    "threshold_terbaik": float(threshold_terbaik),
    "batas_ekstrem_utama_mm": 100,
    "batas_ekstrem_info_mm": 150,
    "horizon_hari": 7,
    "best_params_optuna": study.best_params,
    "roc_auc_cv5_tuned": float(cv_tuned_df["roc_auc"].mean()),
    "pr_auc_cv5_tuned": float(cv_tuned_df["pr_auc"].mean()),
    "tanggal_training": str(pd.Timestamp.now()),
}
with open(os.path.join(MODEL_DIR, "metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)

log(f"\nModel final (hasil tuning) disimpan -> {MODEL_DIR}/xgboost_model_final.pkl (menimpa model Step 5 lama)")
log(f"Metadata diperbarui -> {MODEL_DIR}/metadata.pkl")


# -------------------------------------------------------------
# 6. SIMPAN LAPORAN
# -------------------------------------------------------------
path_report_txt = os.path.join(REPORT_DIR, "05b_laporan_tuning_optuna.txt")
with open(path_report_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

log(f"\nLaporan tuning tersimpan: {path_report_txt}")
print("\nSELESAI.")