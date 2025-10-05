# dashboard/app.py
# Fixed-grid Streamlit dashboard for inference.
# Grid is locked to the Switzerland training grid.
# PV on the LEFT, consumption on the RIGHT, and combined bar chart in that order.

import os
import json
import datetime as dt
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from dotenv import load_dotenv

import folium
from streamlit_folium import st_folium
import meteomatics.api as api
import altair as alt  # <-- for explicit bar order


# =========================
# Configuration (LOCKED GRID)
# =========================
TIMEZONE   = "Europe/Zurich"
MODELS_DIR = "models"

# Switzerland bbox (N, W, S, E) and grid resolution — must match training
BBOX_CH   = (47.89, 5.77, 45.74, 10.65)
RES_LAT   = 0.25
RES_LON   = 0.25
DEFAULT_INTERVAL_HOURS = 1  # user may change the interval hours; grid stays fixed

# Variables & aggregations used in training
PARAMETERS = [
    "global_rad:W",
    "direct_rad:W",
    "diffuse_rad:W",
    "effective_cloud_cover:octas",
    "t_2m:C",
    "relative_humidity_2m:p",
    "wind_speed_10m:ms",
    "precip_1h:mm",
    "snow_depth:cm",
    "fresh_snow_1h:cm",
]

AGG_SPEC = {
    "global_rad:W": ["sum", "mean"],
    "t_2m:C": ["min", "max", "mean"],
    "relative_humidity_2m:p": "mean",
    "wind_speed_10m:ms": "mean",
    "precip_1h:mm": "sum",
    "snow_depth:cm": "mean",
    "effective_cloud_cover:octas": "mean",
}


# =========================
# Helper (single function)
# =========================
def fetch_and_build_daily_features(start_utc: dt.datetime,
                                   end_utc: dt.datetime,
                                   bbox: tuple[float, float, float, float],
                                   tz: str,
                                   res_lat: float,
                                   res_lon: float,
                                   interval: dt.timedelta,
                                   parameters: list[str]) -> pd.DataFrame:
    """
    Fetch Meteomatics grid series for [start_utc, end_utc) on a FIXED bbox/res,
    aggregate to a single LOCAL-day row, and add calendar features.
    """
    load_dotenv()
    user = os.getenv("METEOMATICS_USER")
    pw   = os.getenv("METEOMATICS_PASSWORD")
    if not (user and pw):
        raise RuntimeError("Set METEOMATICS_USER / METEOMATICS_PASSWORD in environment or .env")

    latN, lonW, latS, lonE = bbox

    df = api.query_grid_timeseries(
        start_utc, end_utc, interval,
        parameters,
        latN, lonW, latS, lonE,
        res_lat, res_lon,
        user, pw,
        request_type="POST",
        interp_select="lapse_rate"
    ).reset_index()

    local_day = start_utc.astimezone(ZoneInfo(tz)).date()
    idx = pd.to_datetime([local_day])

    # If API returned nothing, build an empty-but-shaped row (model's imputer can handle NaNs)
    if df.empty:
        out = pd.DataFrame(index=idx)
        for k, v in AGG_SPEC.items():
            if isinstance(v, list):
                for a in v:
                    out[f"{k}_{a}"] = np.nan
            else:
                out[f"{k}_{v}"] = np.nan
        out["month"] = idx.month
        out["dow"] = idx.dayofweek
        out["is_weekend"] = (out["dow"] >= 5).astype(int)
        out.index.name = "date"
        return out

    # Group by LOCAL date (robust around midnight/DST)
    t_local = pd.to_datetime(df["validdate"]).dt.tz_convert(tz)
    df["_date_local"] = t_local.dt.date

    agg_spec_filtered = {k: v for k, v in AGG_SPEC.items() if k in df.columns}
    if not agg_spec_filtered:
        out = pd.DataFrame(index=idx)
        out["month"] = idx.month
        out["dow"] = idx.dayofweek
        out["is_weekend"] = (out["dow"] >= 5).astype(int)
        out.index.name = "date"
        return out

    daily = df.groupby("_date_local").agg(agg_spec_filtered)
    daily.columns = ["_".join(c) if isinstance(c, tuple) else c for c in daily.columns]
    daily.index = pd.to_datetime(daily.index)

    # Force exactly one row for the selected local day (missing -> NaN)
    daily = daily.reindex(idx)

    # Calendar features
    daily["month"] = idx.month
    daily["dow"] = idx.dayofweek
    daily["is_weekend"] = (daily["dow"] >= 5).astype(int)
    daily.index.name = "date"

    return daily


# =========================
# App
# =========================
st.set_page_config(page_title="Energy/PV Forecast — Fixed Switzerland Grid", layout="wide")
st.title("🔆 Energy / PV Forecast — Fixed Switzerland Grid")

# Compute grid details from fixed bbox/res
latN, lonW, latS, lonE = BBOX_CH
center_lat = (latN + latS) / 2.0
center_lon = (lonE + lonW) / 2.0
rows = int(np.floor((latN - latS) / RES_LAT)) + 1
cols = int(np.floor((lonE - lonW) / RES_LON)) + 1

# Top: grid details + date / interval controls
colA, colB, colC, colD = st.columns([1, 1, 1, 1])
with colA:
    st.metric("Center lat", f"{center_lat:.4f}")
with colB:
    st.metric("Center lon", f"{center_lon:.4f}")
with colC:
    st.metric("Rows × Cols", f"{rows} × {cols}")
with colD:
    st.metric("Resolution (lat, lon)", f"{RES_LAT:.2f}°, {RES_LON:.2f}°")

st.caption(f"**BBox (N, W, S, E):** {latN:.4f}, {lonW:.4f}, {latS:.4f}, {lonE:.4f}")

col_map, col_ctrl = st.columns([2, 1])

# Map (read-only view)
with col_map:
    st.subheader("🗺️ Switzerland Training Grid (read-only)")
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="cartodbpositron")
    m.fit_bounds([[latS, lonW], [latN, lonE]])

    folium.Rectangle(
        bounds=[[latS, lonW], [latN, lonE]],
        color="#2A93D5", weight=2, fill=False, dash_array="6,6", tooltip="Training BBox"
    ).add_to(m)

    # Draw grid lines (limited density for performance)
    if rows * cols <= 12000:
        for i in range(rows):
            lat_i = latS + i * RES_LAT
            folium.PolyLine([[lat_i, lonW], [lat_i, lonE]], weight=1, opacity=0.5).add_to(m)
        for j in range(cols):
            lon_j = lonW + j * RES_LON
            folium.PolyLine([[latS, lon_j], [latN, lon_j]], weight=1, opacity=0.5).add_to(m)
    else:
        st.info("Grid too dense to preview all lines; showing only bbox.")

    st_folium(m, height=540, use_container_width=True, key="fixed_map")  # read-only display

# Controls (date + interval + predict)
with col_ctrl:
    st.subheader("📅 Date")
    sel_date = st.date_input("Local Swiss date", value=dt.date.today())

    st.subheader("⏱ Interval")
    interval_hours = st.number_input(
        "Hours", min_value=1, max_value=24,
        value=int(DEFAULT_INTERVAL_HOURS), step=1
    )
    interval = dt.timedelta(hours=int(interval_hours))

    st.subheader("🔮 Predict")
    go = st.button("Run", type="primary")

st.divider()

# Load models (expecting pv_production_gwh and national_consumption_gwh)
targets, models, metas = [], {}, {}
for target_name in ["pv_production_gwh", "national_consumption_gwh"]:
    model_path = Path(MODELS_DIR) / f"{target_name}.joblib"
    meta_path  = Path(MODELS_DIR) / f"{target_name}.meta.json"
    if model_path.exists() and meta_path.exists():
        models[target_name] = joblib.load(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            metas[target_name] = json.load(f)
        targets.append(target_name)

if not targets:
    st.error(f"No models found in `{MODELS_DIR}`. Train models first.")
    st.stop()

# Predict
if go:
    try:
        # Local-day → UTC window
        start_local = dt.datetime(sel_date.year, sel_date.month, sel_date.day, 0, 0, tzinfo=ZoneInfo(TIMEZONE))
        end_local   = start_local + dt.timedelta(days=1)
        start_utc   = start_local.astimezone(dt.timezone.utc)
        end_utc     = end_local.astimezone(dt.timezone.utc)

        with st.spinner("Fetching Meteomatics weather & building features..."):
            features = fetch_and_build_daily_features(
                start_utc=start_utc,
                end_utc=end_utc,
                bbox=BBOX_CH,
                tz=TIMEZONE,
                res_lat=RES_LAT,
                res_lon=RES_LON,
                interval=interval,
                parameters=PARAMETERS,
            )

        if features.shape[0] == 0:
            st.warning("No feature rows for the selected day — cannot predict.")
            st.stop()

        st.subheader("Built Daily Features")
        st.dataframe(features.reset_index())

        # --- Predictions (PV LEFT, Consumption RIGHT) ---
        st.subheader("Predictions")

        PV_KEY  = "pv_production_gwh"
        CON_KEY = "national_consumption_gwh"

        # 1) compute predictions FIRST (no plotting yet)
        y_hat_pv = None
        y_hat_con = None

        if PV_KEY in models:
            meta_pv = metas[PV_KEY]
            X_pv = features.reindex(columns=meta_pv["feature_columns"], fill_value=np.nan)
            y_pred_pv = np.asarray(models[PV_KEY].predict(X_pv)).ravel()
            if y_pred_pv.size > 0:
                y_hat_pv = float(y_pred_pv[0])

        if CON_KEY in models:
            meta_con = metas[CON_KEY]
            X_con = features.reindex(columns=meta_con["feature_columns"], fill_value=np.nan)
            y_pred_con = np.asarray(models[CON_KEY].predict(X_con)).ravel()
            if y_pred_con.size > 0:
                y_hat_con = float(y_pred_con[0])

        # 2) lay out: PV (left) and Consumption (right)
        left_col, right_col = st.columns(2)

        with left_col:
            if y_hat_pv is not None:
                st.metric(label="PV production (GWh)", value=f"{y_hat_pv:.3f}")
            else:
                st.error("PV model returned no prediction.")

        with right_col:
            if y_hat_con is not None:
                st.metric(label="National consumption (GWh)", value=f"{y_hat_con:.3f}")
            else:
                st.error("Consumption model returned no prediction.")

        # 3) combined bar chart UNDERNEATH, fixed order: PV then consumption
        bar_df = pd.DataFrame({
            "target": ["PV production", "National consumption"],
            "key":    [PV_KEY, CON_KEY],  # internal keys (not shown)
            "prediction_gwh": [
                y_hat_pv if y_hat_pv is not None else np.nan,
                y_hat_con if y_hat_con is not None else np.nan
            ],
        })

        order = ["PV production", "National consumption"]
        chart = (
            alt.Chart(bar_df)
            .mark_bar()
            .encode(
                x=alt.X("target:N", sort=order, title=""),
                y=alt.Y("prediction_gwh:Q", title="GWh"),
                tooltip=["target:N", alt.Tooltip("prediction_gwh:Q", format=".3f")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)

        with st.expander("Details"):
            st.write("**BBox (N, W, S, E):**", BBOX_CH)
            st.write("**Grid center (lat, lon):**", (center_lat, center_lon))
            st.write("**Rows × Cols:**", (rows, cols))
            st.write("**Resolution (lat, lon deg):**", (RES_LAT, RES_LON))
            st.write("**Interval:**", interval)
            st.json({k: {
                "val_start_date": metas[k].get("val_start_date"),
                "n_estimators": metas[k].get("n_estimators")
            } for k in targets})

    except Exception as e:
        st.error(f"Prediction failed: {e}")
