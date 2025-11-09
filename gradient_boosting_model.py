import os
import joblib
import hopsworks
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 🔐 Load environment
load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")
if not api_key:
    raise ValueError("❌ Missing HOPSWORKS_API_KEY")

# 🔗 Connect to Hopsworks
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

# 📥 Load data
fg = fs.get_feature_group(name="aqi_data", version=1)
df = fg.read().dropna(subset=["aqi"]).sort_values("datetime")

# 📊 Prepare features
features = ["temp", "humidity", "pressure", "wind_speed", "pm2_5", "pm10",
            "no2", "so2", "o3", "co", "day", "month", "hour"]
target = "aqi"
X, y = df[features], df[target]

# ⏱ TimeSeries K-Fold Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
rmse_list, mae_list, r2_list = [], [], []

print("🔁 Performing Time-Series Cross-Validation...")
for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    rmse_list.append(rmse)
    mae_list.append(mae)
    r2_list.append(r2)
    
    print(f"✅ Fold {fold} — RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

# 📈 Average metrics across folds
print("\n📊 Cross-Validation Results:")
print(f"Average RMSE: {np.mean(rmse_list):.3f} ± {np.std(rmse_list):.3f}")
print(f"Average MAE: {np.mean(mae_list):.3f} ± {np.std(mae_list):.3f}")
print(f"Average R²: {np.mean(r2_list):.3f} ± {np.std(r2_list):.3f}")

# 💾 Train final model on full data
final_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)
final_model.fit(X, y)

os.makedirs("models", exist_ok=True)
path = f"models/GradientBoosting_CV_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"
joblib.dump(final_model, path)
print(f"💾 Final model saved to {path}")
