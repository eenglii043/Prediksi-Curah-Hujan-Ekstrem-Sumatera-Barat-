"""
=============================================================
STEP 6 (FINAL) - KLIMATOLOGI HARMONIK + SARIMA STEPWISE SEARCH (PARALEL)
                 + BACKTEST + ESTIMASI HARIAN 2026-2028
=============================================================
CARA PAKAI:
1. Taruh file ini di folder yang sama dengan folder 03_Hasil/
   (folder tempat file 02_data_siap_modeling.csv berada)
2. pip install joblib   (kalau belum ada)
3. Jalankan: python 06_estimasi_risiko_final.py
4. Tunggu sampai selesai (total sekitar 10-15 menit di komputer 4 core)
=============================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
import time
from joblib import Parallel, delayed

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

# -------------------------------------------------------------
# 0. KONFIGURASI
# -------------------------------------------------------------
INPUT_PATH = "03_Hasil/02_data_siap_modeling.csv"
OUTPUT_DIR = "03_Hasil"
REPORT_DIR = "laporan_hasil"

TANGGAL_BATAS_PENDEK = 7
TARGET_AKHIR_ESTIMASI = "2028-12-31"
PERSENTIL_KLIMATOLOGI = [50, 75, 90, 95]
N_HARMONIK = 4
HOLDOUT_TAHUN = 2
SEASONAL_PERIOD = 12
N_JOBS = -1
MAX_KOMPLEKSITAS = 4
MAXITER_OPTIMIZER = 50

P_MAX, D_MAX, Q_MAX = 2, 1, 2
SP_MAX, SD_MAX, SQ_MAX = 1, 1, 1

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


# -------------------------------------------------------------
# FUNGSI: STEPWISE SEARCH SARIMA
# -------------------------------------------------------------
def fit_best_sarima(series):
    bounds = [P_MAX, D_MAX, Q_MAX, SP_MAX, SD_MAX, SQ_MAX]

    def valid(order_tuple):
        if any(v < 0 or v > b for v, b in zip(order_tuple, bounds)):
            return False
        p, d, q, P, D, Q = order_tuple
        return (p + q + P + Q) <= MAX_KOMPLEKSITAS

    def try_fit(order_tuple):
        p, d, q, P, D, Q = order_tuple
        try:
            model = SARIMAX(
                series, order=(p, d, q), seasonal_order=(P, D, Q, SEASONAL_PERIOD),
                enforce_stationarity=False, enforce_invertibility=False
            )
            fitted = model.fit(disp=False, maxiter=MAXITER_OPTIMIZER, method="lbfgs")
            if fitted.mle_retvals.get("converged", True) is False:
                return None, np.inf
            return fitted, fitted.aic
        except Exception:
            return None, np.inf

    starting_points = [
        (1, 0, 1, 1, 0, 1),
        (0, 1, 1, 0, 1, 1),
        (1, 1, 0, 1, 1, 0),
        (2, 0, 1, 1, 0, 0),
        (1, 0, 0, 0, 1, 1),
    ]
    starting_points = [sp for sp in starting_points if valid(sp)]

    evaluated = {}
    best_order, best_fit, best_aic = None, None, np.inf

    for sp in starting_points:
        fitted, aic = try_fit(sp)
        evaluated[sp] = aic
        if aic < best_aic:
            best_aic, best_fit, best_order = aic, fitted, sp

    if best_order is None:
        return None, None, np.inf

    improved = True
    while improved:
        improved = False
        neighbors = []
        for idx in range(6):
            for delta in (1, -1):
                cand = list(best_order)
                cand[idx] += delta
                cand_tuple = tuple(cand)
                if valid(cand_tuple) and cand_tuple not in evaluated:
                    neighbors.append(cand_tuple)

        for cand_tuple in neighbors:
            fitted, aic = try_fit(cand_tuple)
            evaluated[cand_tuple] = aic
            if aic < best_aic:
                best_aic, best_fit, best_order = aic, fitted, cand_tuple
                improved = True

    return best_fit, best_order, best_aic


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # -------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------
    log("=" * 70)
    log("STEP 6 FINAL - KLIMATOLOGI + SARIMA STEPWISE + BACKTEST + ESTIMASI")
    log("=" * 70)
    log(f"Waktu proses : {pd.Timestamp.now()}")
    log(f"Jumlah CPU core terdeteksi : {os.cpu_count()}")

    df = pd.read_csv(INPUT_PATH, parse_dates=["TANGGAL"])
    station_meta = df.drop_duplicates("STATION_KEY").set_index("STATION_KEY")[
        ["NAMA_STASIUN", "LATITUDE", "LONGITUDE", "ELEVASI_M"]
    ]
    stations = station_meta.index.tolist()
    log(f"Jumlah stasiun : {len(stations)}")

    wide = df.pivot_table(index="TANGGAL", columns="STATION_KEY", values="RAINFALL_MM").sort_index()
    tanggal_terakhir_historis = wide.index.max()
    log(f"Rentang data historis : {wide.index.min().date()} s/d {tanggal_terakhir_historis.date()}")

    # -------------------------------------------------------------
    # 2. KLIMATOLOGI HARMONIK
    # -------------------------------------------------------------
    log("\n" + "=" * 70)
    log(f"STEP 6A - KLIMATOLOGI HARMONIK ({N_HARMONIK} pasang sin/cos)")
    log("=" * 70)

    def build_harmonic_features(doy_series, n_harm=N_HARMONIK, period=365.25):
        X = np.ones((len(doy_series), 1))
        for k in range(1, n_harm + 1):
            X = np.hstack([
                X,
                np.sin(2 * np.pi * k * doy_series.values.reshape(-1, 1) / period),
                np.cos(2 * np.pi * k * doy_series.values.reshape(-1, 1) / period)
            ])
        return X

    df_long = wide.reset_index().melt(id_vars="TANGGAL", var_name="STATION_KEY", value_name="RAINFALL_MM")
    df_long["DOY"] = df_long["TANGGAL"].dt.dayofyear

    doy_full = pd.Series(range(1, 366))
    X_full = build_harmonic_features(doy_full)

    def compute_climatology_for_station(s):
        sub = df_long[(df_long["STATION_KEY"] == s) & (df_long["RAINFALL_MM"].notna())].copy()
        if len(sub) < 365:
            return None
        X = build_harmonic_features(sub["DOY"])
        y = sub["RAINFALL_MM"].values
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        fitted_curve = np.clip(X_full @ coef, 0, None)

        sub["FITTED"] = X @ coef
        sub["RESIDUAL"] = sub["RAINFALL_MM"] - sub["FITTED"]

        rows = []
        for doy in range(1, 366):
            lo, hi = doy - 10, doy + 10
            doy_range = [((d - 1) % 365) + 1 for d in range(lo, hi + 1)]
            resid_window = sub[sub["DOY"].isin(doy_range)]["RESIDUAL"]
            mean_val = fitted_curve[doy - 1]
            row = {"STATION_KEY": s, "DOY": doy, "MEAN_HARMONIK": mean_val, "N_OBS": len(resid_window)}
            for p in PERSENTIL_KLIMATOLOGI:
                if len(resid_window) > 0:
                    row[f"P{p}_HISTORIS"] = max(0, mean_val + np.percentile(resid_window, p) - np.median(resid_window))
                else:
                    row[f"P{p}_HISTORIS"] = mean_val
            rows.append(row)
        return rows

    t0 = time.time()
    klim_results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(compute_climatology_for_station)(s) for s in stations
    )
    klim_percentile_rows = [row for res in klim_results if res is not None for row in res]
    klimatologi = pd.DataFrame(klim_percentile_rows)
    log(f"Klimatologi harmonik selesai untuk {klimatologi['STATION_KEY'].nunique()} stasiun "
        f"({klimatologi.shape[0]:,} baris) dalam {time.time()-t0:.1f} detik.")

    # -------------------------------------------------------------
    # 3. SARIMA STEPWISE SEARCH (PARALEL)
    # -------------------------------------------------------------
    log("\n" + "=" * 70)
    log("STEP 6B - SARIMA STEPWISE SEARCH (paralel per stasiun)")
    log("=" * 70)

    monthly = wide.resample("MS").mean()
    target_akhir = pd.Timestamp(TARGET_AKHIR_ESTIMASI)
    n_bulan_depan = (target_akhir.year - monthly.index.max().year) * 12 + \
                    (target_akhir.month - monthly.index.max().month)
    log(f"Proyeksi ke depan : {n_bulan_depan} bulan")

    def tune_station(s, series):
        if len(series) < 48:
            return s, None, None, "GAGAL_DATA_KURANG"
        best_fit, best_order, best_aic = fit_best_sarima(series)
        if best_fit is None:
            return s, None, None, "GAGAL_KONVERGENSI"
        fc = best_fit.get_forecast(steps=n_bulan_depan)
        fc_mean = fc.predicted_mean.clip(lower=0)
        return s, fc_mean, {"ORDER_pdq": best_order[:3], "SEASONAL_ORDER_PDQ": best_order[3:], "AIC": round(best_aic, 2)}, "OK"

    t0 = time.time()
    tuning_results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(tune_station)(s, monthly[s].dropna()) for s in stations
    )
    log(f"Tuning SARIMA selesai dalam {time.time()-t0:.1f} detik.")

    sarima_forecasts, sarima_order_report, gagal_sarima = {}, [], []
    for s, fc_mean, order_info, status in tuning_results:
        if status == "OK":
            sarima_forecasts[s] = fc_mean
            sarima_order_report.append({"STATION_KEY": s, **order_info})
        else:
            gagal_sarima.append((s, status))

    log(f"SARIMA berhasil di-tuning : {len(sarima_forecasts)} stasiun")
    log(f"SARIMA gagal (fallback rata-rata bulanan historis): {len(gagal_sarima)} stasiun")
    for s, status in gagal_sarima:
        log(f"  - {s}: {status}")

    future_index = pd.date_range(monthly.index.max() + pd.offsets.MonthBegin(1), periods=n_bulan_depan, freq="MS")
    for s, _ in gagal_sarima:
        bulan_avg = monthly[s].groupby(monthly.index.month).mean()
        fc_fallback = pd.Series([bulan_avg.get(d.month, monthly[s].mean()) for d in future_index], index=future_index)
        sarima_forecasts[s] = fc_fallback

    sarima_monthly_df = pd.DataFrame(sarima_forecasts)
    order_report_df = pd.DataFrame(sarima_order_report)

    # -------------------------------------------------------------
    # 4. BACKTEST (PARALEL)
    # -------------------------------------------------------------
    log("\n" + "=" * 70)
    log(f"STEP 6C - BACKTEST (hold-out {HOLDOUT_TAHUN} tahun terakhir, paralel)")
    log("=" * 70)

    cutoff = tanggal_terakhir_historis - pd.DateOffset(years=HOLDOUT_TAHUN)
    monthly_train = monthly[monthly.index <= cutoff]
    monthly_test = monthly[monthly.index > cutoff]
    log(f"Cutoff backtest : {cutoff.date()} | Data test : {len(monthly_test)} bulan")

    def backtest_station(s):
        series_train = monthly_train[s].dropna()
        series_test = monthly_test[s].dropna()
        if len(series_train) < 48 or len(series_test) == 0:
            return None
        best_fit, best_order, _ = fit_best_sarima(series_train)
        if best_fit is None:
            return None
        fc = best_fit.get_forecast(steps=len(series_test)).predicted_mean.clip(lower=0)
        fc_aligned = fc.reindex(series_test.index)
        mae = np.mean(np.abs(fc_aligned - series_test))
        rmse = np.sqrt(np.mean((fc_aligned - series_test) ** 2))
        return {"STATION_KEY": s, "MAE_mm_bulanan": round(mae, 2), "RMSE_mm_bulanan": round(rmse, 2)}

    t0 = time.time()
    backtest_results = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(backtest_station)(s) for s in stations)
    backtest_df = pd.DataFrame([r for r in backtest_results if r is not None])
    log(f"Backtest selesai dalam {time.time()-t0:.1f} detik untuk {len(backtest_df)} stasiun")
    if len(backtest_df) > 0:
        log(f"MAE rata-rata (mm/bulan)  : {backtest_df['MAE_mm_bulanan'].mean():.2f}")
        log(f"RMSE rata-rata (mm/bulan) : {backtest_df['RMSE_mm_bulanan'].mean():.2f}")

    # -------------------------------------------------------------
    # 5. ESTIMASI HARIAN 2026-2028 (vectorized, sederhana, tanpa loop ganda)
    # -------------------------------------------------------------
    log("\n" + "=" * 70)
    log("STEP 6D - ESTIMASI HARIAN (klimatologi harmonik x faktor skala SARIMA)")
    log("=" * 70)

    rata2_bulan_historis = monthly.groupby(monthly.index.month).mean()
    rata2_bulan_historis.index.name = "BULAN"  # pastikan nama index eksplisit, jangan andalkan reset_index() tebak nama

    future_daily = pd.date_range(tanggal_terakhir_historis + pd.Timedelta(days=1), target_akhir, freq="D")
    df_future = pd.DataFrame({"TANGGAL": future_daily})
    df_future["DOY"] = df_future["TANGGAL"].dt.dayofyear.clip(upper=365)
    df_future["BULAN"] = df_future["TANGGAL"].dt.month
    df_future["TAHUN"] = df_future["TANGGAL"].dt.year
    df_future["METODE"] = np.where(
        (df_future["TANGGAL"] - tanggal_terakhir_historis).dt.days <= TANGGAL_BATAS_PENDEK,
        "PREDIKSI_JANGKA_PENDEK", "ESTIMASI_KLIMATOLOGIS"
    )

    sarima_monthly_df_reset = sarima_monthly_df.copy()
    sarima_monthly_df_reset.index.name = "TANGGAL_BULAN"  # set eksplisit, jangan andalkan reset_index() tebak nama
    proyeksi_long = sarima_monthly_df_reset.reset_index()
    proyeksi_long = proyeksi_long.melt(id_vars="TANGGAL_BULAN", var_name="STATION_KEY", value_name="PROYEKSI_BULAN")
    proyeksi_long["TANGGAL_BULAN"] = pd.to_datetime(proyeksi_long["TANGGAL_BULAN"])
    proyeksi_long["TAHUN"] = proyeksi_long["TANGGAL_BULAN"].dt.year
    proyeksi_long["BULAN"] = proyeksi_long["TANGGAL_BULAN"].dt.month
    proyeksi_long = proyeksi_long[["TAHUN", "BULAN", "STATION_KEY", "PROYEKSI_BULAN"]]

    rata2_long = rata2_bulan_historis.reset_index()  # kolom "BULAN" sudah pasti ada krn index.name di-set eksplisit di atas
    rata2_long = rata2_long.melt(id_vars="BULAN", var_name="STATION_KEY", value_name="RATA2_HISTORIS")

    faktor_df = proyeksi_long.merge(rata2_long, on=["BULAN", "STATION_KEY"], how="left")
    faktor_df["FAKTOR_SKALA"] = np.where(
        (faktor_df["RATA2_HISTORIS"] > 0) & faktor_df["PROYEKSI_BULAN"].notna(),
        faktor_df["PROYEKSI_BULAN"] / faktor_df["RATA2_HISTORIS"],
        1.0
    )
    faktor_df["FAKTOR_SKALA"] = faktor_df["FAKTOR_SKALA"].clip(0.3, 3.0)
    faktor_df = faktor_df[["TAHUN", "BULAN", "STATION_KEY", "FAKTOR_SKALA"]]

    stations_df = pd.DataFrame({"STATION_KEY": stations})
    df_future["_key"] = 1
    stations_df["_key"] = 1
    estimasi_df = df_future.merge(stations_df, on="_key").drop(columns="_key")

    estimasi_df = estimasi_df.merge(klimatologi, on=["STATION_KEY", "DOY"], how="left")
    estimasi_df = estimasi_df.merge(faktor_df, on=["TAHUN", "BULAN", "STATION_KEY"], how="left")
    estimasi_df["FAKTOR_SKALA"] = estimasi_df["FAKTOR_SKALA"].fillna(1.0)

    estimasi_df["ESTIMASI_MEAN"] = estimasi_df["MEAN_HARMONIK"] * estimasi_df["FAKTOR_SKALA"]
    estimasi_df["ESTIMASI_P50"] = estimasi_df["P50_HISTORIS"] * estimasi_df["FAKTOR_SKALA"]
    estimasi_df["ESTIMASI_P75"] = estimasi_df["P75_HISTORIS"] * estimasi_df["FAKTOR_SKALA"]
    estimasi_df["ESTIMASI_P90"] = estimasi_df["P90_HISTORIS"] * estimasi_df["FAKTOR_SKALA"]
    estimasi_df["ESTIMASI_P95"] = estimasi_df["P95_HISTORIS"] * estimasi_df["FAKTOR_SKALA"]

    tahun_awal_proyeksi = future_daily.min().year
    estimasi_df["TINGKAT_KETIDAKPASTIAN"] = (
        (estimasi_df["TAHUN"] - tahun_awal_proyeksi) * 0.15 + 0.3
    ).clip(upper=1.0).round(3)

    estimasi_df = estimasi_df.merge(station_meta.reset_index(), on="STATION_KEY", how="left")
    estimasi_df = estimasi_df[[
        "STATION_KEY", "NAMA_STASIUN", "LATITUDE", "LONGITUDE", "ELEVASI_M", "TANGGAL", "METODE",
        "ESTIMASI_MEAN", "ESTIMASI_P50", "ESTIMASI_P75", "ESTIMASI_P90", "ESTIMASI_P95",
        "TINGKAT_KETIDAKPASTIAN"
    ]]

    log(f"Tabel estimasi selesai: {estimasi_df.shape[0]:,} baris")

    # -------------------------------------------------------------
    # 6. SIMPAN OUTPUT
    # -------------------------------------------------------------
    path_klimatologi = os.path.join(OUTPUT_DIR, "06_klimatologi_harmonik_stasiun.csv")
    path_estimasi = os.path.join(OUTPUT_DIR, "06_estimasi_risiko_historis_2026_2028.csv")
    path_order = os.path.join(OUTPUT_DIR, "06_sarima_order_terbaik_per_stasiun.csv")
    path_backtest = os.path.join(OUTPUT_DIR, "06_backtest_akurasi_2tahun.csv")
    path_report_txt = os.path.join(REPORT_DIR, "06_laporan_klimatologi_estimasi.txt")

    klimatologi.to_csv(path_klimatologi, index=False)
    estimasi_df.to_csv(path_estimasi, index=False)
    order_report_df.to_csv(path_order, index=False)
    backtest_df.to_csv(path_backtest, index=False)

    log("\n" + "=" * 70)
    log("OUTPUT TERSIMPAN")
    log("=" * 70)
    log(f"1. Klimatologi harmonik              : {path_klimatologi}")
    log(f"2. Estimasi risiko historis s/d 2028 : {path_estimasi}")
    log(f"3. Order SARIMA terbaik per stasiun  : {path_order}")
    log(f"4. Laporan akurasi (backtest 2 th)   : {path_backtest}")
    log(f"5. Laporan proses (txt)              : {path_report_txt}")

    with open(path_report_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\nSELESAI.")


if __name__ == "__main__":
    main()
