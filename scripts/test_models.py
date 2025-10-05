# scripts/test_models.py
# Load saved models, rebuild validation slice, print metrics, and save prediction figures + CSVs.

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Args (must match training data locations) ---
RAW_CON_PATH = "data/raw/ogd103_stromverbrauch_swissgrid_lv_und_endv.csv"
RAW_PV_PATH  = "data/raw/ogd104_stromproduktion_swissgrid.csv"
WEATHER_DIR  = "data/processed/meteomatics_weather_hourly_2017"
MODELS_DIR   = "models"
REPORTS_DIR  = "reports"
TARGETS      = ["pv_production_gwh", "national_consumption_gwh"]

Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

# --- Load & prep energy consumption ---
df_con = (
    pd.read_csv(RAW_CON_PATH)
      .rename(columns={
          "Datum": "date",
          "Landesverbrauch_GWh": "national_consumption_gwh",
          "Endverbrauch_GWh": "end_consumption_gwh",
      })
      .dropna(subset=["date"])
)
df_con["date"] = pd.to_datetime(df_con["date"])

# --- Load & prep PV production ---
df_pv = (
    pd.read_csv(RAW_PV_PATH)
      .rename(columns={
          "Datum": "date",
          "Energietraeger": "energy_carrier",
          "Produktion_GWh": "production_gwh",
      })
)
df_pv = (
    df_pv[df_pv["energy_carrier"].str.lower().eq("photovoltaik")]
      .drop(columns=["energy_carrier"])
)
df_pv["date"] = pd.to_datetime(df_pv["date"])
df_pv = (
    df_pv.groupby("date", as_index=False)["production_gwh"].sum()
         .rename(columns={"production_gwh": "pv_production_gwh"})
)

# --- Load & aggregate weather ---
weather_files = [
    os.path.join(WEATHER_DIR, f)
    for f in os.listdir(WEATHER_DIR)
    if f.endswith(".csv")
]
if not weather_files:
    raise FileNotFoundError(f"No CSV files found in {WEATHER_DIR}")

df_w = pd.concat(
    [pd.read_csv(p, parse_dates=["validdate"]) for p in weather_files],
    ignore_index=True
)

df_w["date"] = pd.to_datetime(df_w["validdate"]).dt.date

agg_spec = {
    "global_rad:W": ["sum", "mean"],
    "t_2m:C": ["min", "max", "mean"],
    "relative_humidity_2m:p": "mean",
    "wind_speed_10m:ms": "mean",
    "precip_1h:mm": "sum",
    "snow_depth:cm": "mean",
    "effective_cloud_cover:octas": "mean",
}
agg_spec = {k: v for k, v in agg_spec.items() if k in df_w.columns}

df_w = df_w.groupby("date").agg(agg_spec)
df_w.columns = ["_".join(c) if isinstance(c, tuple) else c for c in df_w.columns]
df_w = df_w.reset_index().rename(columns={"date": "date"})
df_w["date"] = pd.to_datetime(df_w["date"])
df_w["month"] = df_w["date"].dt.month
df_w["dow"] = df_w["date"].dt.dayofweek
df_w["is_weekend"] = (df_w["dow"] >= 5).astype(int)

# --- Join all on date ---
df_all = (
    df_w.merge(df_con, on="date", how="left")
        .merge(df_pv, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
)
df = df_all.copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

import joblib
from sklearn.metrics import mean_absolute_error, r2_score

# --- Evaluate per target with saved metadata ---
for target in TARGETS:
    model_path = Path(MODELS_DIR) / f"{target}.joblib"
    meta_path  = Path(MODELS_DIR) / f"{target}.meta.json"

    if not model_path.exists() or not meta_path.exists():
        print(f"[skip] Missing model or metadata for {target}")
        continue

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_columns = meta["feature_columns"]
    val_start_date  = pd.to_datetime(meta["val_start_date"])

    dft = df.dropna(subset=[target]).reset_index(drop=True)
    if dft.empty:
        print(f"[skip] No data for target={target}")
        continue

    # Recreate validation slice used in training
    mask_val = dft["date"] >= val_start_date
    X_val = dft.loc[mask_val, feature_columns]
    y_val = dft.loc[mask_val, target]
    t_val = dft.loc[mask_val, "date"]

    model = joblib.load(model_path)
    y_pred = model.predict(X_val)

    mae = float(mean_absolute_error(y_val, y_pred))
    r2  = float(r2_score(y_val, y_pred))
    print(f"{target}: MAE={mae:.3f} | R2={r2:.3f} | val from {val_start_date.date()} ({len(X_val)} samples)")

    # Save figure
    plt.figure(figsize=(11, 4))
    plt.plot(t_val, y_val.values, label="actual")
    plt.plot(t_val, y_pred, label="pred", alpha=0.85)
    plt.title(f"{target} — MAE={mae:.3f}, R2={r2:.3f}")
    plt.xlabel("date"); plt.ylabel("GWh"); plt.legend(); plt.tight_layout()

    fig_path = Path(REPORTS_DIR) / f"{target}_pred_vs_actual.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved figure: {fig_path}")

    # Save predictions CSV
    out_csv = Path(REPORTS_DIR) / f"{target}_predictions.csv"
    pd.DataFrame({"date": t_val, "actual": y_val.values, "pred": y_pred}).to_csv(out_csv, index=False)
    print(f"Saved predictions: {out_csv}")
