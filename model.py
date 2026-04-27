import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from data import get_sensor_readings
from features import build_feature_grid

FEATURES = ["no2", "road_density", "elevation", "wind_speed", "hour_of_day"]

def find_nearest_cell(grid_df, lat, lon):
    dist = np.sqrt((grid_df["lat"] - lat)**2 + (grid_df["lon"] - lon)**2)
    return dist.idxmin()

def train_and_predict():
    print("Building feature grid...")
    grid_df = build_feature_grid()

    print("Loading sensor readings...")
    sensors = get_sensor_readings()

    train_rows = []
    for _, sensor in sensors.iterrows():
        idx = find_nearest_cell(grid_df, sensor["lat"], sensor["lon"])
        row = grid_df.loc[idx].copy()
        row["aqi"] = sensor["aqi"]
        row["sensor_name"] = sensor["name"]
        train_rows.append(row)
    train_df = pd.DataFrame(train_rows)

    X_train = train_df[FEATURES]
    y_train = train_df["aqi"]

    print(f"Training XGBoost on {len(X_train)} sensor locations...")
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    mae = mean_absolute_error(y_train, train_preds)
    print(f"Training MAE: {mae:.2f} AQI units")

    print("Predicting AQI for all grid cells...")
    X_all = grid_df[FEATURES]
    grid_df["predicted_aqi"] = model.predict(X_all)
    grid_df["predicted_aqi"] = grid_df["predicted_aqi"].clip(0, 500).round(1)

    importance = dict(zip(FEATURES, model.feature_importances_))
    print("\nFeature importances:")
    for f, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {f}: {imp:.3f}")

    grid_df.to_csv("predictions.csv", index=False)
    print(f"\nSaved predictions.csv ({len(grid_df)} cells)")
    return grid_df, sensors, model

if __name__ == "__main__":
    train_and_predict()