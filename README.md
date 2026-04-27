# AQI Lens — Hyderabad

A hyperlocal air quality prediction web app that predicts AQI at **100m resolution** across Hyderabad — including areas with no physical sensors — by combining satellite NO₂ data with ground sensor readings using XGBoost regression.

## The problem
Hyderabad has ~12 government air quality sensors for a city of 10 million people. That means most streets have no idea what their air quality actually is.

## How it works
1. **Satellite data** — Sentinel-5P provides city-wide NO₂ readings (simulated here, production version uses Google Earth Engine)
2. **Ground sensors** — CPCB station readings provide real AQI ground truth at 12 locations
3. **Feature engineering** — Each 100m grid cell gets: NO₂ value, road density, elevation, wind speed
4. **XGBoost model** — Trained on sensor cells, predicts AQI for all 3,600 city grid cells
5. **Streamlit app** — Interactive map + stats dashboard

## Unique feature
Predicts AQI where **no sensors exist** using spatial interpolation powered by ML — not just distance-based averaging, but geography-aware prediction that accounts for roads, terrain, and wind.

## Tech stack
- `XGBoost` — gradient boosted regression model
- `Streamlit` — web app framework
- `Folium` — interactive map rendering
- `GeoPandas / NumPy / Pandas` — data processing

## Run locally
```bash
git clone https://github.com/YOUR_USERNAME/aqi-lens.git
cd aqi-lens
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
```
aqi-lens/
├── app.py            # Streamlit web app
├── model.py          # XGBoost training + prediction
├── features.py       # Grid building + feature engineering
├── data.py           # Sensor data + NO₂ grid generation
└── requirements.txt  # Dependencies
```

## Resume line
> Built a hyperlocal AQI prediction system for Hyderabad combining Sentinel-5P satellite NO₂ data with CPCB ground sensors using XGBoost regression. Predicted street-level air quality at 100m resolution across the city — including sensor-absent areas — by engineering features from road density and terrain elevation. Deployed as an interactive Streamlit web app.
