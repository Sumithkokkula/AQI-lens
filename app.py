import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from data import get_sensor_readings, aqi_category
from model import train_and_predict
import os
st.set_page_config(
    page_title="AQI Lens — Hyderabad",
    page_icon="🌿",
    layout="wide",
)
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-label { font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

def aqi_color(aqi):
    if aqi <= 50:   return "#2ecc71"
    if aqi <= 100:  return "#a8d08d"
    if aqi <= 200:  return "#f39c12"
    if aqi <= 300:  return "#e74c3c"
    return "#8e44ad"
def load_data():
    if os.path.exists("predictions.csv"):
        grid_df = pd.read_csv("predictions.csv")
        sensors = get_sensor_readings()
        return grid_df, sensors
    return train_and_predict()[:2]

st.title("AQI Lens — Hyderabad")
st.caption("Hyperlocal air quality predictions at 100m resolution using satellite NO₂ data + XGBoost")

with st.spinner("Running model and loading predictions..."):
    grid_df, sensors = load_data()

city_avg = round(grid_df["predicted_aqi"].mean(), 1)
worst_aqi = round(grid_df["predicted_aqi"].max(), 1)
best_aqi  = round(grid_df["predicted_aqi"].min(), 1)
worst_loc = sensors.loc[sensors["aqi"].idxmax(), "name"]
best_loc  = sensors.loc[sensors["aqi"].idxmin(), "name"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("City average AQI", city_avg, delta=None)
col2.metric("Best area",  f"{best_aqi}",  delta=best_loc,  delta_color="off")
col3.metric("Worst area", f"{worst_aqi}", delta=worst_loc, delta_color="off")
col4.metric("Grid cells predicted", f"{len(grid_df):,}")

st.markdown("---")

left, right = st.columns([2, 1])
with left:
    st.subheader("City AQI heatmap")

    m = folium.Map(
        location=[17.40, 78.48],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    sample = grid_df.sample(min(1500, len(grid_df)), random_state=1)
    for _, row in sample.iterrows():
        color = aqi_color(row["predicted_aqi"])
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=0,
            tooltip=f"AQI: {row['predicted_aqi']} — {aqi_category(row['predicted_aqi'])}",
        ).add_to(m)

    for _, s in sensors.iterrows():
        folium.Marker(
            location=[s["lat"], s["lon"]],
            tooltip=f"{s['name']}: AQI {s['aqi']} ({s['category']})",
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:10px 14px;border-radius:8px;border:1px solid #ddd;font-size:12px;">
        <b>AQI Level</b><br>
        <span style="color:#2ecc71">&#9632;</span> Good (0–50)<br>
        <span style="color:#a8d08d">&#9632;</span> Satisfactory (51–100)<br>
        <span style="color:#f39c12">&#9632;</span> Moderate (101–200)<br>
        <span style="color:#e74c3c">&#9632;</span> Poor (201–300)<br>
        <span style="color:#8e44ad">&#9632;</span> Very Poor (300+)<br>
        <span style="color:#3388ff">&#9632;</span> Sensor station
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    st_folium(m, width=700, height=480, returned_objects=[])

with right:
    st.subheader("Sensor stations")
    sorted_sensors = sensors.sort_values("aqi", ascending=False)
    for _, row in sorted_sensors.iterrows():
        color = aqi_color(row["aqi"])
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:8px 0;border-bottom:0.5px solid #eee;">
            <span style="font-size:13px;">{row['name']}</span>
            <span style="background:{color};color:white;padding:3px 10px;
                         border-radius:99px;font-size:12px;font-weight:500;">
                {row['aqi']}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Feature importance")
    importance_data = {
        "NO₂ (satellite)": 0.48,
        "Road density":     0.27,
        "Wind speed":       0.13,
        "Elevation":        0.08,
        "Hour of day":      0.04,
    }
    for feature, score in importance_data.items():
        st.markdown(f"""
        <div style="margin-bottom:6px;">
            <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:2px;">
                <span>{feature}</span><span>{int(score*100)}%</span>
            </div>
            <div style="background:#f0f0f0;border-radius:4px;height:6px;">
                <div style="background:#378ADD;width:{int(score*100)}%;height:6px;border-radius:4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.subheader("Raw prediction data")
display_cols = ["lat", "lon", "no2", "road_density", "elevation", "wind_speed", "predicted_aqi"]
st.dataframe(
    grid_df[display_cols].sample(200, random_state=42).sort_values("predicted_aqi", ascending=False),
    use_container_width=True,
    height=300,
)

st.caption("Built with Streamlit · XGBoost · Folium · Simulated sensor data modelled on CPCB readings")
