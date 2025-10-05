import os
import datetime as dt
from dateutil.relativedelta import relativedelta
import pandas as pd
import meteomatics.api as api
from dotenv import load_dotenv


def fetch_meteomatics_grid_timeseries(
    start: dt.datetime,
    end: dt.datetime,
    *,
    out_dir: str,
    bbox: tuple[float, float, float, float] = (47.89, 5.77, 45.74, 10.65),
    res_lat: float = 0.25,
    res_lon: float = 0.25,
    interval: dt.timedelta = dt.timedelta(hours=1),
    parameters: list[str] | None = None,
    tz: str = "Europe/Zurich",
):
    """
    Fetch Meteomatics GRID time series data in monthly chunks and save each chunk as CSV.

    Parameters
    ----------
    start, end : datetime
        UTC datetime range (start inclusive, end exclusive).
    out_dir : str
        Directory to save monthly CSV files.
    bbox : (N, W, S, E)
        Bounding box.
    res_lat, res_lon : float
        Grid resolution.
    interval : timedelta
        Sampling interval (e.g. 1h, 1d).
    parameters : list[str]
        Meteomatics variable codes.
    tz : str
        Local timezone for derived columns (hour, dow, etc.).

    Notes
    -----
    • Credentials are read from environment variables:
        METEOMATICS_USER, METEOMATICS_PASSWORD
    • Each monthly chunk is saved as: YYYYMMDD_YYYYMMDD.csv
    • Adds local time features: hour, dow, is_weekend, month, day, ts_local
    """

    # Load credentials
    load_dotenv()
    user = os.getenv("METEOMATICS_USER")
    pw   = os.getenv("METEOMATICS_PASSWORD")
    if not (user and pw):
        raise RuntimeError("Set METEOMATICS_USER / METEOMATICS_PASSWORD in .env or environment")

    # Default variables if none passed
    if parameters is None:
        parameters = [
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

    LAT_N, LON_W, LAT_S, LON_E = bbox
    os.makedirs(out_dir, exist_ok=True)

    # Helper to iterate month by month
    def month_chunks(a, b):
        cur = a
        while cur < b:
            nxt = cur + relativedelta(months=1)
            yield cur, min(nxt, b)
            cur = nxt

    # Fetch data month by month
    for s, e in month_chunks(start, end):
        print(f"Fetching {s:%Y-%m-%d} - {e:%Y-%m-%d}")
        df = api.query_grid_timeseries(
            s, e, interval,
            parameters,
            LAT_N, LON_W, LAT_S, LON_E,
            res_lat, res_lon,
            user, pw,
            request_type="POST",
            interp_select="lapse_rate"
        ).reset_index()

        # Add local time columns
        t = df["validdate"].dt.tz_convert(tz)
        df["hour"] = t.dt.hour
        df["dow"] = t.dt.dayofweek
        df["is_weekend"] = (t.dt.dayofweek >= 5).astype(int)
        df["month"] = t.dt.month
        df["day"] = t.dt.day
        df["ts_local"] = t.dt.tz_localize(None)

        # Save monthly file
        path = os.path.join(out_dir, f"{s:%Y%m%d}_{e:%Y%m%d}.csv")
        df.to_csv(path, index=False)
        print(f"Saved: {path}")
