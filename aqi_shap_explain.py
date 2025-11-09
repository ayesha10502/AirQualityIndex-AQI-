# shap_explain_hopsworks.py
import os
import joblib
import hopsworks
import pandas as pd
import shap
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pathlib import Path

# ----------------------------
# LOAD ENV VARIABLES
# ----------------------------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    raise ValueError("❌ HOPSWORKS_API_KEY not found in .env file.")

# ----------------------------
# CONFIG
# ----------------------------
FEATURE_GROUP_NAME = "aqi_data"
FEATURE_GROUP_VERSION = 1
MODEL_NAME = "aqi_best_model"

FEATURE_COLS = [
    "temp", "humidity", "pressure", "wind_speed",
    "pm2_5", "pm10", "no2", "so2", "o3", "co",
    "day", "month", "hour"
]

# ----------------------------
# CONNECT TO HOPSWORKS
# ----------------------------
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# Load feature group
fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
df = fg.read()
df = df.dropna(subset=["aqi"])  # Ensure target exists
X = df[FEATURE_COLS]

print("✅ Features loaded from Hopsworks.")

# ----------------------------
# LOAD MODEL
# ----------------------------
mr = project.get_model_registry()
models = mr.get_models(name=MODEL_NAME)
if not models:
    raise RuntimeError("❌ No models found in Hopsworks registry.")
latest_model = models[-1]
model_path = latest_model.download()

# If model folder contains multiple files, pick .pkl or .joblib
if os.path.isdir(model_path):
    files = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith(".pkl") or f.endswith(".joblib")]
    if not files:
        raise RuntimeError("❌ No model artifact found in downloaded folder.")
    model_file = files[0]
else:
    model_file = model_path

model = joblib.load(model_file)
print("✅ Model loaded from Hopsworks.")

# ----------------------------
# SHAP EXPLAINABILITY
# ----------------------------
print("🔹 Computing SHAP values...")
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# ----------------------------
# FEATURE IMPORTANCE PLOT
# ----------------------------
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=True)
# Optional: save plot
# plt.savefig("shap_summary.png")

print("✅ SHAP summary plot generated!")
