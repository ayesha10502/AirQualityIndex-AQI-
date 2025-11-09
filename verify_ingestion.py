import hopsworks
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

fg = fs.get_feature_group(name="aqi_data", version=1)
df = fg.read()

latest = pd.to_datetime(df["datetime_str"]).max()
print(f"📊 Total records: {len(df)}")
print(f"🕒 Latest timestamp: {latest}")

if latest < (datetime.utcnow() - timedelta(hours=24)):
    raise ValueError("❌ Verification failed: No new data found in the last 24 hours.")
else:
    print("✅ Data ingestion is up-to-date.")
