import numpy as np
import pandas as pd
from data import get_no2_grid, HYD_LAT_MIN, HYD_LAT_MAX, HYD_LON_MIN, HYD_LON_MAX, GRID_SIZE

np.random.seed(42)

ROADS = [
    {"lat": 17.385, "lon": 78.486, "density": 0.95},
    {"lat": 17.445, "lon": 78.456, "density": 0.90},
    {"lat": 17.406, "lon": 78.556, "density": 0.85},
    {"lat": 17.430, "lon": 78.415, "density": 0.80},
    {"lat": 17.485, "lon": 78.414, "density": 0.75},
    {"lat": 17.347, "lon": 78.552, "density": 0.82},
    {"lat": 17.372, "lon": 78.361, "density": 0.40},
    {"lat": 17.416, "lon": 78.435, "density": 0.25},
]

def build_feature_grid():
    lat_mesh, lon_mesh, no2_grid = get_no2_grid()

    road_density = np.zeros((GRID_SIZE, GRID_SIZE))
    for r in ROADS:
        dist = np.sqrt((lat_mesh - r["lat"])**2 + (lon_mesh - r["lon"])**2)
        road_density += r["density"] * np.exp(-dist / 0.03)
    road_density = np.clip(road_density, 0, 1)

    elevation = 510 + 40 * np.sin(lat_mesh * 60) * np.cos(lon_mesh * 60)
    elevation += np.random.normal(0, 5, elevation.shape)

    wind_speed = 12 + 8 * np.cos((lat_mesh - 17.4) * 30) + np.random.normal(0, 2, lat_mesh.shape)
    wind_speed = np.clip(wind_speed, 1, 35)

    hour_of_day = 10

    rows = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rows.append({
                "lat":          round(lat_mesh[i, j], 6),
                "lon":          round(lon_mesh[i, j], 6),
                "no2":          round(no2_grid[i, j], 6),
                "road_density": round(road_density[i, j], 4),
                "elevation":    round(elevation[i, j], 2),
                "wind_speed":   round(wind_speed[i, j], 2),
                "hour_of_day":  hour_of_day,
                "grid_i":       i,
                "grid_j":       j,
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = build_feature_grid()
    print(f"Grid built: {len(df)} cells")
    print(df.head())
    