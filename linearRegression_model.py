import os
import joblib
import hopsworks
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 🔐 Load environment
load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")
if not api_key:
    raise ValueError("❌ Missing HOPSWORKS_API_KEY")

# 🔗 Connect to Hopsworks
print("🔐 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

# 📥 Load data
print("📦 Fetching data from Hopsworks...")
fg = fs.get_feature_group(name="aqi_data", version=1)
df = fg.read()
df = df.dropna(subset=["aqi"]).sort_values("datetime")

print(f"✅ Data fetched: {df.shape[0]} rows × {df.shape[1]} columns")

# 📊 Prepare features
features = ["temp", "humidity", "pressure", "wind_speed", "pm2_5", "pm10", "no2", "so2", "o3", "co", "day", "month", "hour"]
target = "aqi"
X, y = df[features], df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# 🧠 Train model
print("🧠 Training Linear Regression...")
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 📈 Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"✅ Linear Regression — RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

# 💾 Save model
os.makedirs("models", exist_ok=True)
path = f"models/LinearRegression_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"
joblib.dump(model, path)
print(f"💾 Model saved to {path}")
