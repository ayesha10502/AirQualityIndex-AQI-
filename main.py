import os
import time
from datetime import datetime, timedelta
import requests
import pandas as pd
import hopsworks
from dotenv import load_dotenv

# ---- LOAD ENV -----
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not OPENWEATHER_API_KEY or not HOPSWORKS_API_KEY:
    raise ValueError("API keys not set. Ensure they exist in .env or GitHub Secrets.")

# ---- CONFIG -----
LAT, LON = 34.0151, 71.5249  # Peshawar
CITY = "Peshawar"
HISTORY_DAYS = 180

FG_NAME = "aqi_data"
FG_VERSION = 1
PRIMARY_KEY = "datetime_str"

# ---- CONNECT TO HOPSWORKS -----
print("🔐 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# ---- GET OR CREATE FEATURE GROUP -----
try:
    feature_group = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
    print("🔍 Feature group found.")
    existing_df = feature_group.read()
    if not existing_df.empty:
        last_datetime = pd.to_datetime(existing_df["datetime_str"]).max()
        print(f"🕒 Last record in Hopsworks: {last_datetime}")
        start_date = last_datetime + timedelta(hours=1)
    else:
        print("ℹ️ Feature group is empty. Fetching historical data...")
        start_date = datetime.utcnow() - timedelta(days=HISTORY_DAYS)
except Exception:
    print("🆕 Feature group not found. Creating new one...")
    start_date = datetime.utcnow() - timedelta(days=HISTORY_DAYS)
    schema = [
        {"name": "datetime", "type": "timestamp"},
        {"name": "temp", "type": "double"},
        {"name": "humidity", "type": "bigint"},
        {"name": "pressure", "type": "bigint"},
        {"name": "wind_speed", "type": "double"},
        {"name": "aqi", "type": "bigint"},
        {"name": "co", "type": "double"},
        {"name": "no2", "type": "double"},
        {"name": "o3", "type": "double"},
        {"name": "so2", "type": "double"},
        {"name": "pm2_5", "type": "double"},
        {"name": "pm10", "type": "double"},
        {"name": "datetime_str", "type": "string"},
        {"name": "day", "type": "int"},
        {"name": "month", "type": "int"},
        {"name": "hour", "type": "int"},
        {"name": "aqi_change_rate", "type": "double"},
    ]
    feature_group = fs.create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        primary_key=[PRIMARY_KEY],
        description=f"Weather + Air Quality data for {CITY}",
        features=schema,
        online_enabled=True,
    )

end_date = datetime.utcnow()
print(f"📆 Fetching data from {start_date} to {end_date}")

# ---- FETCH HISTORICAL DATA -----
def fetch_openweather_data(start_date, end_date):
    weather_data, pollution_data = [], []
    current_date = start_date

    while current_date <= end_date:
        start_ts = int(current_date.timestamp())
        end_ts = int((current_date + timedelta(days=1)).timestamp())

        weather_url = (
            f"https://history.openweathermap.org/data/2.5/history/city?"
            f"lat={LAT}&lon={LON}&type=hour&start={start_ts}&end={end_ts}"
            f"&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        pollution_url = (
            f"http://api.openweathermap.org/data/2.5/air_pollution/history?"
            f"lat={LAT}&lon={LON}&start={start_ts}&end={end_ts}"
            f"&appid={OPENWEATHER_API_KEY}"
        )

        try:
            w_resp = requests.get(weather_url, timeout=15)
            p_resp = requests.get(pollution_url, timeout=15)
            print(f"📅 {current_date.date()} Weather: {w_resp.status_code}, Pollution: {p_resp.status_code}")

            w_data = w_resp.json().get("list", []) if w_resp.status_code == 200 else []
            p_data = p_resp.json().get("list", []) if p_resp.status_code == 200 else []

            for hour in w_data:
                main = hour.get("main", {})
                wind = hour.get("wind", {})
                weather_data.append({
                    "datetime": datetime.utcfromtimestamp(hour["dt"]),
                    "temp": float(main.get("temp", 0)),
                    "humidity": int(main.get("humidity", 0)),
                    "pressure": int(main.get("pressure", 0)),
                    "wind_speed": float(wind.get("speed", 0)),
                })

            for entry in p_data:
                components = entry.get("components", {})
                pollution_data.append({
                    "datetime": datetime.utcfromtimestamp(entry["dt"]),
                    "aqi": int(entry.get("main", {}).get("aqi", 0)),
                    "co": float(components.get("co", 0)),
                    "no2": float(components.get("no2", 0)),
                    "o3": float(components.get("o3", 0)),
                    "so2": float(components.get("so2", 0)),
                    "pm2_5": float(components.get("pm2_5", 0)),
                    "pm10": float(components.get("pm10", 0)),
                })

        except Exception as e:
            print(f"⚠️ Error fetching {current_date.date()}: {e}")

        current_date += timedelta(days=1)
        time.sleep(1)

    weather_df = pd.DataFrame(weather_data)
    pollution_df = pd.DataFrame(pollution_data)

    # Ensure datetime exists
    if "datetime" not in weather_df.columns:
        weather_df["datetime"] = pd.to_datetime([])
    if "datetime" not in pollution_df.columns:
        pollution_df["datetime"] = pd.to_datetime([])

    return weather_df, pollution_df

# ---- FETCH REAL-TIME DATA -----
def fetch_realtime_data():
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}&units=metric"
    pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}"

    try:
        w_resp = requests.get(weather_url, timeout=10)
        p_resp = requests.get(pollution_url, timeout=10)

        if w_resp.status_code != 200 or p_resp.status_code != 200:
            print(f"⚠️ Real-time fetch failed: Weather {w_resp.status_code}, Pollution {p_resp.status_code}")
            return pd.DataFrame()

        w = w_resp.json().get("main", {})
        wind = w_resp.json().get("wind", {})
        p = p_resp.json().get("list", [{}])[0]
        comp = p.get("components", {})

        now = datetime.utcnow()
        return pd.DataFrame([{
            "datetime": now,
            "temp": float(w.get("temp", 0)),
            "humidity": int(w.get("humidity", 0)),
            "pressure": int(w.get("pressure", 0)),
            "wind_speed": float(wind.get("speed", 0)),
            "aqi": int(p.get("main", {}).get("aqi", 0)),
            "co": float(comp.get("co", 0)),
            "no2": float(comp.get("no2", 0)),
            "o3": float(comp.get("o3", 0)),
            "so2": float(comp.get("so2", 0)),
            "pm2_5": float(comp.get("pm2_5", 0)),
            "pm10": float(comp.get("pm10", 0)),
            "datetime_str": now.strftime("%Y-%m-%d %H:%M:%S"),
            "day": now.day,
            "month": now.month,
            "hour": now.hour,
            "aqi_change_rate": 0.0,
        }])

    except Exception as e:
        print(f"⚠️ Error fetching real-time data: {e}")
        return pd.DataFrame()

# ---- FETCH + MERGE -----
weather_df, pollution_df = fetch_openweather_data(start_date, end_date)

if weather_df.empty and pollution_df.empty:
    print("⚠️ No historical data fetched. Fetching real-time only...")
    merged_df = pd.DataFrame()
else:
    if weather_df.empty or pollution_df.empty:
        merged_df = pd.DataFrame()
    else:
        merged_df = pd.merge_asof(
            weather_df.sort_values("datetime"),
            pollution_df.sort_values("datetime"),
            on="datetime",
            direction="nearest",
            tolerance=pd.Timedelta("1h"),
        )

# ---- ADD REAL-TIME DATA -----
realtime_df = fetch_realtime_data()
if not realtime_df.empty:
    merged_df = pd.concat([merged_df, realtime_df], ignore_index=True)

# ---- CHECK FOR DUPLICATES -----
if 'existing_df' in locals() and not existing_df.empty:
    merged_df = merged_df[~merged_df["datetime_str"].isin(existing_df["datetime_str"])]

if merged_df.empty:
    print("⚠️ No new data to insert today. Exiting.")
    exit(0)

# ---- FIX SCHEMA TYPES -----
merged_df["day"] = merged_df["day"].astype("int32")
merged_df["month"] = merged_df["month"].astype("int32")
merged_df["hour"] = merged_df["hour"].astype("int32")
merged_df["wind_speed"] = merged_df["wind_speed"].astype("float")
merged_df["aqi_change_rate"] = merged_df["aqi_change_rate"].astype("float")

# ---- SAVE + UPLOAD -----
csv_file = f"aqi_data_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
merged_df.to_csv(csv_file, index=False)
print(f"💾 Saved new data locally: {csv_file}")

print(f"⬆️ Inserting {len(merged_df)} new records into Hopsworks...")
feature_group.insert(merged_df)
print("✅ Data successfully inserted into Hopsworks!")