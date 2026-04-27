import numpy as np
import pandas as pd

np.random.seed(42)

HYD_LAT_MIN, HYD_LAT_MAX = 17.30, 17.50
HYD_LON_MIN, HYD_LON_MAX = 78.40, 78.60
GRID_SIZE = 60

SENSOR_LOCATIONS = [
    {"name": "Punjagutta",        "lat": 17.4239, "lon": 78.4738, "base_aqi": 145},
    {"name": "Nacharam",          "lat": 17.4062, "lon": 78.5562, "base_aqi": 168},
    {"name": "Bollaram",          "lat": 17.4725, "lon": 78.4152, "base_aqi": 112},
    {"name": "Sanathnagar",       "lat": 17.4444, "lon": 78.4397, "base_aqi": 138},
    {"name": "Somajiguda",        "lat": 17.4284, "lon": 78.4569, "base_aqi": 152},
    {"name": "Kukatpally",        "lat": 17.4849, "lon": 78.4138, "base_aqi": 128},
    {"name": "LB Nagar",          "lat": 17.3469, "lon": 78.5522, "base_aqi": 156},
    {"name": "ECIL",              "lat": 17.4625, "lon": 78.5633, "base_aqi": 144},
    {"name": "Bahadurpally",      "lat": 17.5387, "lon": 78.4674, "base_aqi": 98},
    {"name": "Mallapur",          "lat": 17.4075, "lon": 78.5703, "base_aqi": 162},
    {"name": "KBR Park",          "lat": 17.4156, "lon": 78.4347, "base_aqi": 48},
    {"name": "Gandipet",          "lat": 17.3722, "lon": 78.3614, "base_aqi": 55},
]

def get_sensor_readings():
    rows = []
    for s in SENSOR_LOCATIONS:
        noise = np.random.normal(0, 8)
        aqi = max(20, min(300, s["base_aqi"] + noise))
        no2 = aqi * 0.0038 + np.random.normal(0, 0.002)
        rows.append({
            "name":     s["name"],
            "lat":      s["lat"],
            "lon":      s["lon"],
            "aqi":      round(aqi, 1),
            "no2":      round(max(0, no2), 5),
            "category": aqi_category(aqi),
        })
    return pd.DataFrame(rows)

def aqi_category(aqi):
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Satisfactory"
    if aqi <= 200:  return "Moderate"
    if aqi <= 300:  return "Poor"
    return "Very Poor"

def get_no2_grid():
    lats = np.linspace(HYD_LAT_MIN, HYD_LAT_MAX, GRID_SIZE)
    lons = np.linspace(HYD_LON_MIN, HYD_LON_MAX, GRID_SIZE)
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    no2 = np.zeros((GRID_SIZE, GRID_SIZE))
    for s in SENSOR_LOCATIONS:
        dist = np.sqrt((lat_mesh - s["lat"])**2 + (lon_mesh - s["lon"])**2)
        influence = s["base_aqi"] * 0.004 * np.exp(-dist / 0.04)
        no2 += influence
    no2 += np.random.normal(0, 0.003, no2.shape)
    no2 = np.clip(no2, 0.001, 1.5)
    return lat_mesh, lon_mesh, no2