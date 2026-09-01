import json
import pandas as pd

# Load dataset
df = pd.read_csv("data/ambulance_data.csv", low_memory=False)

# Convert timestamps and compute response time in minutes
df["received_dttm"] = pd.to_datetime(df["received_dttm"])
df["on_scene_dttm"] = pd.to_datetime(df["on_scene_dttm"])
df["response_time_minutes"] = (df["on_scene_dttm"] - df["received_dttm"]).dt.total_seconds() / 60.0

# Filter out bad timestamps and extreme outliers (keep 0.5 to 60 minutes)
clean_df = df[(df["response_time_minutes"] >= 0.5) & (df["response_time_minutes"] <= 60)].copy()

# Calculate clean medians by priority
median_stats = clean_df.groupby("priority")["response_time_minutes"].median().to_dict()

# Save clean stats to JSON inside the data folder
with open("data/summary_stats.json", "w") as f:
    json.dump({"priority_stats": median_stats}, f, indent=4)

print("Successfully updated summary_stats.json with cleaned medians!")