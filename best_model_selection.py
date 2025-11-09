import os
import joblib
import hopsworks
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

# ============================================================== 
# 🔐 Load Environment
# ============================================================== 
load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")
if not api_key:
    raise ValueError("❌ Missing HOPSWORKS_API_KEY in .env")

# ============================================================== 
# 🔗 Connect to Hopsworks
# ============================================================== 
print("🔐 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()
mr = project.get_model_registry()

# ============================================================== 
# 📥 Fetch Data from Feature Store
# ============================================================== 
print("📦 Fetching latest AQI data from Hopsworks...")
fg = fs.get_feature_group(name="aqi_data", version=1)
df = fg.read()

if df.empty:
    raise ValueError("❌ No data found in Feature Store. Please run ingestion first!")

df = df.dropna(subset=["aqi"]).sort_values("datetime")

features = [
    "temp", "humidity", "pressure", "wind_speed",
    "pm2_5", "pm10", "no2", "so2", "o3", "co",
    "day", "month", "hour"
]
target = "aqi"

X = df[features]
y = df[target]

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=42
)

# ============================================================== 
# 📂 Load Models from models/
# ============================================================== 
models_dir = "models"
if not os.path.exists(models_dir):
    raise FileNotFoundError("❌ No 'models/' directory found. Train models first.")

model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
if not model_files:
    raise FileNotFoundError("❌ No .pkl model files found in 'models/'. Train some models first.")

print(f"🧩 Found {len(model_files)} models to compare:")
for f in model_files:
    print(f"  - {f}")

# ============================================================== 
# ⚙️ Evaluate Each Model
# ============================================================== 
results = []
for file in model_files:
    model_path = os.path.join(models_dir, file)
    try:
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model File": file,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

        print(f"✅ Evaluated {file} — RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

    except Exception as e:
        print(f"⚠️ Failed to evaluate {file}: {e}")

# ============================================================== 
# 🏆 Select Best Model
# ============================================================== 
results_df = pd.DataFrame(results)
if results_df.empty:
    raise ValueError("❌ No models successfully evaluated.")

best_row = results_df.loc[results_df["RMSE"].idxmin()]
best_model_file = best_row["Model File"]
best_model_path = os.path.join(models_dir, best_model_file)
best_model = joblib.load(best_model_path)

print("\n🏆 Best Model Selected:")
print(best_row)

# ============================================================== 
# 💾 Save Best Model Locally as best_model.pkl
# ============================================================== 
final_best_path = os.path.join(models_dir, "best_model.pkl")
joblib.dump(best_model, final_best_path)
print(f"\n💾 Saved best model as {final_best_path}")

# ============================================================== 
# 📤 Register Best Model in Hopsworks
# ============================================================== 
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
model_name = "aqi_best_model"

model_metrics = {
    "RMSE": float(best_row["RMSE"]),
    "MAE": float(best_row["MAE"]),
    "R2": float(best_row["R2"])
}

# Create model schema based on current dataset
input_schema = Schema(X)
output_schema = Schema(y.to_frame())
model_schema = ModelSchema(input_schema, output_schema)


print("\n📦 Registering best model in Hopsworks Model Registry...")
model_meta = mr.python.create_model(
    name=model_name,
    version=None,  # version will auto-increment
    metrics=model_metrics,
    description=f"Best AQI regression model selected automatically ({best_model_file})",
    input_example=None,
    model_schema=model_schema
)

# Save/upload model artifact
model_meta.save(final_best_path)
print(f"✅ Model '{model_name}' successfully registered in Hopsworks!")

# ============================================================== 
# 📊 Summary
# ============================================================== 
print("\n📊 Model Comparison Summary:")
print(results_df.sort_values("RMSE").reset_index(drop=True))

print("\n🎯 Best model saved locally and pushed to Hopsworks Registry successfully!")
print("✅ Done.")
