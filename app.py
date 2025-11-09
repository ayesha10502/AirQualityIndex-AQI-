# app.py — 3-Day Daily AQI Forecast + Historical Lookup (Peshawar)

import os
import joblib
import hopsworks
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt

# ===============================================================
# 🔐 Load Environment Variables
# ===============================================================
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
if not HOPSWORKS_API_KEY:
    st.error("❌ Missing HOPSWORKS_API_KEY in .env file.")
    st.stop()

# ===============================================================
# ⚙️ Configuration
# ===============================================================
FEATURE_GROUP_NAME = "aqi_data"
FEATURE_GROUP_VERSION = 1
MODEL_NAME = "aqi_best_model"
CITY = "Peshawar"

FEATURE_COLS = [
    "temp", "humidity", "pressure", "wind_speed",
    "pm2_5", "pm10", "no2", "so2", "o3", "co",
    "day", "month", "hour"
]

# ===============================================================
# 🎨 Streamlit UI Configuration
# ===============================================================
st.set_page_config(page_title=f"AQI Forecast - {CITY}", layout="wide")

st.markdown(
    f"""
    <div style="text-align:center; padding: 10px 0;">
        <h1 style="color:#2E86C1;">🌤️ 3-Day Air Quality Forecast — {CITY}</h1>
        <p style="font-size:18px; color:#555;">
            Using Hopsworks Feature Store and Machine Learning Model Registry
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ===============================================================
# 🧩 Connect to Hopsworks
# ===============================================================
@st.cache_resource
def get_hopsworks_project():
    return hopsworks.login(api_key_value=HOPSWORKS_API_KEY)

@st.cache_data(ttl=600)
def load_feature_data():
    project = get_hopsworks_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    return fg.read()

@st.cache_resource
def load_latest_model():
    project = get_hopsworks_project()
    mr = project.get_model_registry()
    models = mr.get_models(name=MODEL_NAME)
    if not models:
        raise RuntimeError("❌ No models found in Hopsworks registry.")
    latest = models[-1]
    model_path = latest.download()

    if os.path.isdir(model_path):
        files = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith(".pkl") or f.endswith(".joblib")]
        if not files:
            raise RuntimeError("❌ No model artifact found in downloaded folder.")
        model_file = files[0]
    else:
        model_file = model_path

    return joblib.load(model_file)

# ===============================================================
# 📦 Load Data & Model
# ===============================================================
with st.spinner("🔗 Connecting to Hopsworks and loading model..."):
    df = load_feature_data()
    model = load_latest_model()

if df.empty:
    st.error("❌ No data found in Hopsworks feature group.")
    st.stop()

df = df.sort_values("datetime").dropna(subset=["aqi"])

# ===============================================================
# 🧮 AQI Category
# ===============================================================
def get_aqi_category(aqi_value):
    if aqi_value <= 1.5:
        return "🟢 Good (0–50)"
    elif aqi_value <= 2.5:
        return "🟡 Fair / Moderate (51–100)"
    elif aqi_value <= 3.5:
        return "🟠 Moderate / Unhealthy (101–150)"
    elif aqi_value <= 4.5:
        return "🔴 Poor / Unhealthy (151–200)"
    else:
        return "🟣 Very Poor / Hazardous (201–500)"

# ===============================================================
# 📅 Predict Next 3 Days (Daily Aggregated)
# ===============================================================
st.markdown("## 📅 Predicted AQI for the Next 3 Days")

last_row = df.tail(1).copy()
base_time = pd.to_datetime(last_row["datetime"].values[0])
recent_window = df.tail(24)

future_preds = []
for i in range(1, 4):  # next 3 days
    next_date = base_time + timedelta(days=i)

    # average last 24 hours as base
    feature_means = recent_window[[
        "temp", "humidity", "pressure", "wind_speed",
        "pm2_5", "pm10", "no2", "so2", "o3", "co"
    ]].mean()

    next_features = pd.DataFrame([feature_means])
    next_features["day"] = next_date.day
    next_features["month"] = next_date.month
    next_features["hour"] = 12  # midday for daily estimate

    # slight random variation for natural trend
    for col in feature_means.index:
        next_features[col] *= np.random.uniform(0.97, 1.03)

    pred = model.predict(next_features[FEATURE_COLS])[0]
    category = get_aqi_category(pred)

    future_preds.append({
        "Date": next_date.strftime("%Y-%m-%d"),
        "Predicted AQI": round(pred, 2),
        "Category": category,
        "PM2.5": round(float(next_features["pm2_5"]), 2),
        "PM10": round(float(next_features["pm10"]), 2),
        "NO2": round(float(next_features["no2"]), 2),
        "SO2": round(float(next_features["so2"]), 2),
        "O3": round(float(next_features["o3"]), 2),
        "CO": round(float(next_features["co"]), 2)
    })

pred_df = pd.DataFrame(future_preds)
st.dataframe(pred_df, use_container_width=True)

# ===============================================================
# 📊 Visualization
# ===============================================================
st.markdown("## 📈 AQI Forecast Trend (Next 3 Days)")

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(pred_df["Date"], pred_df["Predicted AQI"], color="#2E86C1", alpha=0.8)
ax.set_title(f"Daily AQI Forecast — {CITY}", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Predicted AQI")
ax.grid(axis="y", linestyle="--", alpha=0.5)
st.pyplot(fig)

# ===============================================================
# 🕒 Historical AQI Lookup
# ===============================================================
st.markdown("## 🔍 Check Historical AQI")

selected_date = st.date_input("Select a date to view historical AQI:", value=datetime.utcnow().date() - timedelta(days=1))
day_data = df[df["datetime"].dt.date == selected_date]

if not day_data.empty:
    st.success(f"Showing hourly AQI for {selected_date}")
    st.line_chart(day_data.set_index("datetime")["aqi"])
else:
    st.warning(f"No data available for {selected_date}")

# ===============================================================
# 🧾 AQI Category Reference
# ===============================================================
st.markdown("## 🧾 AQI Category Reference")

category_table = pd.DataFrame({
    "AQI (1–5)": [1, 2, 3, 4, 5],
    "Category": [
        "🟢 Good",
        "🟡 Fair / Moderate",
        "🟠 Moderate / Unhealthy ",
        "🔴 Poor / Unhealthy",
        "🟣 Very Poor / Hazardous"
    ],
    "Equivalent AQI (0–500)": ["0–50", "51–100", "101–150", "151–200", "201–500"]
})
st.table(category_table)

# ===============================================================
# 📍 Footer
# ===============================================================
st.markdown(
    """
    <hr>
    <div style="text-align:center; color:gray; font-size:14px;">
        🚀 Data and model fetched from <b>Hopsworks Feature Store</b> and <b>Model Registry</b>.<br>
        Built by <b>Ayesha Jamil</b>.
    </div>
    """,
    unsafe_allow_html=True
)
