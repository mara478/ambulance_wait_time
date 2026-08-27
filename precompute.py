import os
import json
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "prediction_history.csv")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "data", "summary_stats.json")

print("Processing historical dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Check for existing response_time_minutes or calculate it
if "response_time_minutes" not in df.columns:
    if "received_dttm" in df.columns and "response_dttm" in df.columns:
        df["received_dttm"] = pd.to_datetime(df["received_dttm"], errors="coerce")
        df["response_dttm"] = pd.to_datetime(df["response_dttm"], errors="coerce")
        df["response_time_minutes"] = (df["response_dttm"] - df["received_dttm"]).dt.total_seconds() / 60.0
    else:
        raise KeyError(f"Could not find required time columns in CSV. Available columns are: {list(df.columns)}")

# Ensure hour column exists
if "hour" not in df.columns and "received_dttm" in df.columns:
    df["received_dttm"] = pd.to_datetime(df["received_dttm"], errors="coerce")
    df["hour"] = df["received_dttm"].dt.hour
elif "hour" not in df.columns:
    df["hour"] = 12  # Fallback default if hour is missing

# Filter valid response times
df = df[df["response_time_minutes"] > 0].copy()

# Fill missing categorical values
for col in ["station_area", "neighborhoods_analysis_boundaries", "zipcode_of_incident"]:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna("UNKNOWN").str.strip()
    else:
        df[col] = "UNKNOWN"

stats = {
    "global_mean": float(df["response_time_minutes"].mean()),
    "station_mean": df.groupby("station_area")["response_time_minutes"].mean().to_dict(),
    "neighborhood_mean": df.groupby("neighborhoods_analysis_boundaries")["response_time_minutes"].mean().to_dict(),
    "zip_mean": df.groupby("zipcode_of_incident")["response_time_minutes"].mean().to_dict(),
    "hour_mean": df.groupby("hour")["response_time_minutes"].mean().to_dict(),
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(stats, f)

print("✅ Saved summary_stats.json successfully!")
