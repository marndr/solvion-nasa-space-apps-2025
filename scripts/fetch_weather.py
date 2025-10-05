import datetime as dt
import os
from dotenv import load_dotenv
from src.weather_core import fetch_meteomatics_grid_timeseries

# Load env
load_dotenv()

# --- Define parameters ---
START = dt.datetime(2018, 1, 1)
END   = dt.datetime(2019, 1, 1)
INTERVAL = dt.timedelta(hours=1)

BBOX    = (47.89, 5.77, 45.74, 10.65)
RES_LAT = 0.25
RES_LON = 0.25

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

OUT_DIR = "data/processed/meteomatics_weather_hourly_2018"
TIMEZONE = "Europe/Zurich"

# --- Fetch data ---
fetch_meteomatics_grid_timeseries(
    start=START,
    end=END,
    out_dir=OUT_DIR,
    bbox=BBOX,
    res_lat=RES_LAT,
    res_lon=RES_LON,
    interval=INTERVAL,
    parameters=PARAMETERS,
    tz=TIMEZONE,
)
