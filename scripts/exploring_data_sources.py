import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import zipfile
from io import StringIO

# Test coordinates (Zurich, Switzerland)
LATITUDE = 47.3769
LONGITUDE = 8.5417

# NASA API
def get_nasa_power_data():
    print("Get weather data from NASA POWER API")
    
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    parameters = [
        "ALLSKY_SFC_SW_DWN",
        "T2M",
        "WS10M",
        "RH2M",
        "CLOUD_AMT"
    ]
    
    params = {
        "parameters": ",".join(parameters),
        "community": "RE",
        "longitude": LONGITUDE,
        "latitude": LATITUDE,
        "start": "20241001",
        "end": "20241007",
        "format": "JSON"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        print("NASA POWER API Success")
        return data
    else:
        print(f"Error: {response.status_code}")
        return None

# Execute request
nasa_data = get_nasa_power_data()
print(nasa_data)


# Meteo API
def get_meteoswiss_collections():
    print("Get MeteoSwiss collections from STAC API")
    
    url = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-smn"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        print("MeteoSwiss Collections Success")
        return data
    else:
        print(f"Error: {response.status_code}")
        return None

def get_meteoswiss_stations():
    print("Get MeteoSwiss weather station items")
    
    url = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-smn/items"
    params = {"limit": 10}
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        print("MeteoSwiss Stations Success")
        return data
    else:
        print(f"Error: {response.status_code}")
        return None

def get_meteoswiss_csv_data():
    print("Download actual CSV data from MeteoSwiss station")
    
    csv_url = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/abo/ogd-smn_abo_h_now.csv"
    
    response = requests.get(csv_url)
    
    if response.status_code == 200:
        # Parse CSV content
        df = pd.read_csv(StringIO(response.text), sep=';')
        print("MeteoSwiss CSV Data Success")
        return df
    else:
        print(f"Error: {response.status_code}")
        return None

# Execute requests
meteo_collections = get_meteoswiss_collections()
print(meteo_collections)
meteo_stations = get_meteoswiss_stations()
print(meteo_stations)
meteo_csv = get_meteoswiss_csv_data()
print(meteo_csv)


def get_openmeteo_data():
    print("Get weather forecast from Open-Meteo API")
    
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        'latitude': LATITUDE,
        'longitude': LONGITUDE,
        'hourly': 'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,shortwave_radiation,cloud_cover',
        'timezone': 'Europe/Zurich',
        'forecast_days': 3
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        print("Open-Meteo Success")
        return data
    else:
        print(f"Error: {response.status_code}")
        return None

# Execute request
openmeteo_data = get_openmeteo_data()
print(openmeteo_data)


# Copernicus
def get_cds_data():
    print("Get ERA5 data from Copernicus Climate Data Store")
    
    # Requires: pip install cdsapi
    # Requires: API key from https://cds.climate.copernicus.eu/
    
    import cdsapi
    
    c = cdsapi.Client()  # Reads API key from ~/.cdsapirc
    
    request_params = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'surface_solar_radiation_downwards',
            '2m_temperature', 
            'total_cloud_cover',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind'
        ],
        'year': '2024',
        'month': '10',
        'day': ['01', '02', '03', '04', '05', '06', '07'],
        'time': [
            '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
        ],
        'area': [48, 8, 47, 9]
    }
    
    # Downloads NetCDF file
    c.retrieve('reanalysis-era5-single-levels', request_params, 'era5_data.nc')
    
    # Read with xarray
    import xarray as xr
    zip_path = "era5_data.nc"   # but this is actually a ZIP!
    with zipfile.ZipFile(zip_path, "r") as z:
        # List contents
        print(z.namelist())
        
        # Extract the real netCDF file
        nc_filename = z.namelist()[0]  # usually only one inside
        z.extract(nc_filename, ".")
    
    # Now open the extracted NC file
    dataset = xr.open_dataset(nc_filename, engine="h5netcdf")
    
    return dataset

print(get_cds_data())