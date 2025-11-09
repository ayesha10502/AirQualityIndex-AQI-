import os
import joblib
import hopsworks
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
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

# 📊 Prepare features
features = ["temp", "humidity", "pressure", "wind_speed", "pm2_5", "pm10", "no2", "so2", "o3", "co", "day", "month", "hour"]
target = "aqi"
X, y = df[features], df[target]

# ⚙️ Define model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

# 🔁 5-Fold Cross-Validation
print("🔁 Performing 5-Fold Cross-Validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = []
mae_scores = []
r2_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    rmse_scores.append(rmse)
    mae_scores.append(mae)
    r2_scores.append(r2)

    print(f"✅ Fold {fold} — RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

# 📊 Average results
print("\n📈 Cross-Validation Results (5-Folds):")
print(f"Average RMSE: {np.mean(rmse_scores):.3f} ± {np.std(rmse_scores):.3f}")
print(f"Average MAE: {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
print(f"Average R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

# 🧠 Train final model on all data
print("\n🧠 Training final model on full dataset...")
model.fit(X, y)

# 💾 Save model
os.makedirs("models", exist_ok=True)
path = f"models/RandomForest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"
joblib.dump(model, path)
print(f"💾 Model saved to {path}")
