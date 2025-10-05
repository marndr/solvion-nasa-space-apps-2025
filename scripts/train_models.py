# scripts/train_models.py
# Train and save models (one per target) using the last 20% of time as validation.

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

# --- Args (edit once; reused elsewhere) ---
RAW_CON_PATH = "data/raw/ogd103_stromverbrauch_swissgrid_lv_und_endv.csv"
RAW_PV_PATH  = "data/raw/ogd104_stromproduktion_swissgrid.csv"
WEATHER_DIR  = "data/processed/meteomatics_weather_hourly_2017"  # directory with chunked Meteomatics CSVs
MODELS_DIR   = "models"
VAL_FRACTION = 0.20
RANDOM_STATE = 42
N_ESTIMATORS = 300
TARGETS = ["pv_production_gwh", "national_consumption_gwh"]

Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

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

# --- Load & aggregate weather (all chunk files in WEATHER_DIR) ---
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

# date from timestamp (daily rollup)
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
# keep only available columns
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

# --- Build feature list (drop date + targets) ---
drop_cols = set(["date"]) | set(TARGETS)
feature_cols = [c for c in df.columns if c not in drop_cols]

# --- Train per target and save ---
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

for target in TARGETS:
    dft = df.dropna(subset=[target]).reset_index(drop=True)
    if dft.empty:
        print(f"[skip] No data for target={target}")
        continue

    X = dft[feature_cols]
    y = dft[target]
    t = dft["date"]

    n = len(dft)
    n_test = max(1, int(VAL_FRACTION * n))
    val_start_date = t.iloc[-n_test]  # inclusive

    X_train, X_val = X.iloc[:-n_test], X.iloc[-n_test:]
    y_train, y_val = y.iloc[:-n_test], y.iloc[-n_test:]
    t_val = t.iloc[-n_test:]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    mae = float(mean_absolute_error(y_val, y_pred))
    r2  = float(r2_score(y_val, y_pred))
    print(f"{target}: MAE={mae:.3f} | R2={r2:.3f} | val from {val_start_date.date()} ({n_test} samples)")

    # Save model + metadata
    model_path = Path(MODELS_DIR) / f"{target}.joblib"
    meta_path  = Path(MODELS_DIR) / f"{target}.meta.json"

    joblib.dump(model, model_path)

    metadata = {
        "target": target,
        "feature_columns": feature_cols,     # preserve order
        "val_start_date": val_start_date.strftime("%Y-%m-%d"),
        "val_fraction": VAL_FRACTION,
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "n_samples": n,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

print(f"Saved models and metadata to: {MODELS_DIR}")
