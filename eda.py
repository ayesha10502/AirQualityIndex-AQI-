import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

# ---- LOAD ENV -----
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY not set. Ensure it exists in .env or GitHub Secrets.")

# ---- CONFIG -----
FG_NAME = "aqi_data"
FG_VERSION = 1

# ---- CONNECT TO HOPSWORKS -----
print("🔐 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# ---- FETCH FEATURE GROUP DATA -----
try:
    feature_group = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
    df = feature_group.read()
    if df.empty:
        print("⚠️ Feature group is empty.")
        exit(0)
    print(f"✅ Fetched {len(df)} records from feature group '{FG_NAME}'.")
except Exception as e:
    print(f"⚠️ Error fetching feature group: {e}")
    exit(1)

# ---- BASIC EDA -----
print("\n🔹 Data Info:")
print(df.info())

print("\n🔹 First 5 rows:")
print(df.head())

print("\n🔹 Descriptive statistics (numeric columns):")
numeric_cols = df.select_dtypes(include=["number"]).columns
print(df[numeric_cols].describe())

print("\n🔹 Check for missing values:")
print(df.isna().sum())

print("\n🔹 Data types:")
print(df.dtypes)
