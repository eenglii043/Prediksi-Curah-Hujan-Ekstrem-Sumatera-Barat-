"""
=============================================================
STEP 6 (v2) - KLIMATOLOGI HARMONIK + SARIMA AUTO-TUNING + BACKTEST
Input : 03_Hasil/02_data_siap_modeling.csv (dari Step 2)
Output: klimatologi harmonik, estimasi risiko historis 2026-2028,
        laporan akurasi (backtest 2 tahun terakhir)
=============================================================
"""

import pandas as pd
import numpy as np
import os
import itertools
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX

# -------------------------------------------------------------
# 0. KONFIGURASI
# -------------------------------------------------------------
INPUT_PATH = "03_Hasil/02_data_siap_modeling.csv"
OUTPUT_DIR = "03_Hasil"
REPORT_DIR = "laporan_hasil"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TANGGAL_BATAS_PENDEK = 7
TARGET_AKHIR_ESTIMASI = "2028-12-31"
PERSENTIL_KLIMATOLOGI = [50, 75, 90, 95]
N_HARMONIK = 4                 # jumlah pasangan sin/cos (semakin besar semakin fleksibel kurva musiman)
HOLDOUT_TAHUN = 2               # backtest 2 tahun terakhir

# Grid search SARIMA (dibatasi supaya tetap feasible untuk 128 stasiun x 2 fit @ tiap stasiun)
P_RANGE = [0, 1, 2]
D_RANGE = [0, 1]
Q_RANGE = [0, 1, 2]
SP_RANGE = [0, 1]
SD_RANGE = [0, 1]
SQ_RANGE = [0, 1]
SEASONAL_PERIOD = 12

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
log("=" * 70)
log("STEP 6 v2 - KLIMATOLOGI HARMONIK + SARIMA AUTO-TUNING + BACKTEST")
log("=" * 70)
log(f"Waktu proses : {pd.Timestamp.now()}")

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
# 2. KLIMATOLOGI HARMONIK (regresi Fourier per stasiun)
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

klim_curve_rows = []      # kurva harmonik per (stasiun, doy 1-365) -> mean
klim_percentile_rows = []  # residual -> persentil per (stasiun, doy)
harmonic_coefs = {}

doy_full = pd.Series(range(1, 366))
X_full = build_harmonic_features(doy_full)

for s in stations:
    sub = df_long[(df_long["STATION_KEY"] == s) & (df_long["RAINFALL_MM"].notna())].copy()
    if len(sub) < 365:
        continue
    X = build_harmonic_features(sub["DOY"])
    y = sub["RAINFALL_MM"].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    harmonic_coefs[s] = coef

    fitted_curve = X_full @ coef
    fitted_curve = np.clip(fitted_curve, 0, None)

    # residual aktual - kurva harmonik, dikelompokkan per DOY (window +/-10 hari) utk persentil
    sub["FITTED"] = X @ coef
    sub["RESIDUAL"] = sub["RAINFALL_MM"] - sub["FITTED"]

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
        klim_percentile_rows.append(row)

klimatologi = pd.DataFrame(klim_percentile_rows)
log(f"Klimatologi harmonik selesai untuk {len(harmonic_coefs)} stasiun ({klimatologi.shape[0]:,} baris).")


# -------------------------------------------------------------
# 3. SARIMA AUTO-TUNING (grid search AIC) PER STASIUN, BULANAN
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 6B - SARIMA AUTO-TUNING (grid search AIC, agregasi bulanan)")
log("=" * 70)

monthly = wide.resample("MS").mean()
target_akhir = pd.Timestamp(TARGET_AKHIR_ESTIMASI)
n_bulan_depan = (target_akhir.year - monthly.index.max().year) * 12 + \
                (target_akhir.month - monthly.index.max().month)

param_grid = list(itertools.product(P_RANGE, D_RANGE, Q_RANGE, SP_RANGE, SD_RANGE, SQ_RANGE))
log(f"Ukuran grid search per stasiun : {len(param_grid)} kombinasi")
log(f"Proyeksi ke depan : {n_bulan_depan} bulan")

def fit_best_sarima(series):
    """Grid search kombinasi order berdasarkan AIC terendah. Return (fitted_model, order_terbaik, aic_terbaik)."""
    best_aic = np.inf
    best_fit = None
    best_order = None
    for (p, d, q, P, D, Q) in param_grid:
        if p == 0 and q == 0 and P == 0 and Q == 0:
            continue  # skip model kosong/tidak informatif
        try:
            model = SARIMAX(
                series, order=(p, d, q), seasonal_order=(P, D, Q, SEASONAL_PERIOD),
                enforce_stationarity=False, enforce_invertibility=False
            )
            fitted = model.fit(disp=False)
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_fit = fitted
                best_order = (p, d, q, P, D, Q)
        except Exception:
            continue
    return best_fit, best_order, best_aic

sarima_forecasts = {}
sarima_conf_int = {}
sarima_order_report = []
gagal_sarima = []

for i, s in enumerate(stations):
    series = monthly[s].dropna()
    if len(series) < 48:  # minimal 4 tahun data bulanan utk grid seasonal
        gagal_sarima.append(s)
        continue
    best_fit, best_order, best_aic = fit_best_sarima(series)
    if best_fit is None:
        gagal_sarima.append(s)
        continue
    fc = best_fit.get_forecast(steps=n_bulan_depan)
    fc_mean = fc.predicted_mean.clip(lower=0)
    fc_ci = fc.conf_int(alpha=0.2)  # interval 80%
    sarima_forecasts[s] = fc_mean
    sarima_conf_int[s] = fc_ci
    sarima_order_report.append({
        "STATION_KEY": s, "ORDER_pdq": best_order[:3], "SEASONAL_ORDER_PDQ": best_order[3:], "AIC": round(best_aic, 2)
    })
    if (i + 1) % 20 == 0:
        log(f"  ... {i+1}/{len(stations)} stasiun selesai di-tuning")

log(f"\nSARIMA berhasil di-tuning : {len(sarima_forecasts)} stasiun")
log(f"SARIMA gagal/data kurang (fallback rata-rata bulanan historis): {len(gagal_sarima)} stasiun")

future_index = pd.date_range(monthly.index.max() + pd.offsets.MonthBegin(1), periods=n_bulan_depan, freq="MS")
for s in gagal_sarima:
    bulan_avg = monthly[s].groupby(monthly.index.month).mean()
    fc_fallback = pd.Series([bulan_avg.get(d.month, monthly[s].mean()) for d in future_index], index=future_index)
    sarima_forecasts[s] = fc_fallback

sarima_monthly_df = pd.DataFrame(sarima_forecasts)
order_report_df = pd.DataFrame(sarima_order_report)


# -------------------------------------------------------------
# 4. BACKTEST: HOLD-OUT 2 TAHUN TERAKHIR
# -------------------------------------------------------------
log("\n" + "=" * 70)
log(f"STEP 6C - BACKTEST (hold-out {HOLDOUT_TAHUN} tahun terakhir)")
log("=" * 70)

cutoff = tanggal_terakhir_historis - pd.DateOffset(years=HOLDOUT_TAHUN)
monthly_train = monthly[monthly.index <= cutoff]
monthly_test = monthly[monthly.index > cutoff]
n_test_bulan = len(monthly_test)
log(f"Cutoff backtest : {cutoff.date()} | Data test : {n_test_bulan} bulan")

backtest_rows = []
for s in stations:
    series_train = monthly_train[s].dropna()
    series_test = monthly_test[s].dropna()
    if len(series_train) < 48 or len(series_test) == 0:
        continue
    best_fit, best_order, _ = fit_best_sarima(series_train)
    if best_fit is None:
        continue
    fc = best_fit.get_forecast(steps=len(series_test)).predicted_mean.clip(lower=0)
    fc_aligned = fc.reindex(series_test.index)
    mae = np.mean(np.abs(fc_aligned - series_test))
    rmse = np.sqrt(np.mean((fc_aligned - series_test) ** 2))
    backtest_rows.append({"STATION_KEY": s, "MAE_mm_bulanan": round(mae, 2), "RMSE_mm_bulanan": round(rmse, 2)})

backtest_df = pd.DataFrame(backtest_rows)
if len(backtest_df) > 0:
    log(f"Backtest selesai untuk {len(backtest_df)} stasiun")
    log(f"MAE rata-rata (mm/bulan)  : {backtest_df['MAE_mm_bulanan'].mean():.2f}")
    log(f"RMSE rata-rata (mm/bulan) : {backtest_df['RMSE_mm_bulanan'].mean():.2f}")
else:
    log("Backtest tidak menghasilkan data (kemungkinan data historis kurang dari 2+4 tahun)")


# -------------------------------------------------------------
# 5. ESTIMASI HARIAN 2026-2028: KLIMATOLOGI HARMONIK x FAKTOR SKALA SARIMA (CI-based)
# -------------------------------------------------------------
log("\n" + "=" * 70)
log("STEP 6D - ESTIMASI HARIAN (klimatologi harmonik x faktor skala SARIMA)")
log("=" * 70)

rata2_bulan_historis = monthly.groupby(monthly.index.month).mean()
future_daily = pd.date_range(tanggal_terakhir_historis + pd.Timedelta(days=1), target_akhir, freq="D")
klim_lookup = klimatologi.set_index(["STATION_KEY", "DOY"])

estimasi_rows = []
for d in future_daily:
    doy = min(d.dayofyear, 365)
    bulan = d.month
    is_jangka_pendek = (d - tanggal_terakhir_historis).days <= TANGGAL_BATAS_PENDEK
    metode = "PREDIKSI_JANGKA_PENDEK" if is_jangka_pendek else "ESTIMASI_KLIMATOLOGIS"

    for s in stations:
        try:
            klim = klim_lookup.loc[(s, doy)]
        except KeyError:
            continue

        mask = (sarima_monthly_df.index.year == d.year) & (sarima_monthly_df.index.month == bulan)
        proyeksi_val = sarima_monthly_df.loc[mask, s].iloc[0] if mask.any() else np.nan
        rata2_bulan = rata2_bulan_historis.loc[bulan, s] if bulan in rata2_bulan_historis.index else np.nan

        if s in sarima_conf_int and mask.any():
            ci = sarima_conf_int[s][mask.values]
            lebar_ci = float(ci.iloc[0, 1] - ci.iloc[0, 0]) if len(ci) > 0 else 0.0
            ketidakpastian = min(1.0, lebar_ci / (rata2_bulan + 1e-6)) if rata2_bulan else 0.5
        else:
            ketidakpastian = 0.5  # fallback: ketidakpastian sedang

        faktor = (proyeksi_val / rata2_bulan) if (rata2_bulan and rata2_bulan > 0 and not np.isnan(proyeksi_val)) else 1.0
        faktor = np.clip(faktor, 0.3, 3.0)

        estimasi_rows.append({
            "STATION_KEY": s, "TANGGAL": d, "METODE": metode,
            "ESTIMASI_MEAN": klim["MEAN_HARMONIK"] * faktor,
            "ESTIMASI_P50": klim["P50_HISTORIS"] * faktor,
            "ESTIMASI_P75": klim["P75_HISTORIS"] * faktor,
            "ESTIMASI_P90": klim["P90_HISTORIS"] * faktor,
            "ESTIMASI_P95": klim["P95_HISTORIS"] * faktor,
            "TINGKAT_KETIDAKPASTIAN": round(ketidakpastian, 3),
        })

estimasi_df = pd.DataFrame(estimasi_rows)
estimasi_df = estimasi_df.merge(station_meta.reset_index(), on="STATION_KEY", how="left")
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