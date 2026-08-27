"""
Preprocessing utilities for ambulance response-time prediction.

The trained model expects 27 features. This module builds those
features from the incoming request and historical training data.
"""

import os
import numpy as np
import pandas as pd

VALID_PRIORITIES = ["1", "2", "3", "E", "I"]
VALID_UNIT_TYPES = ["MEDIC"]
VALID_HOURS = list(range(24))

VALID_DAYS_OF_WEEK = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "prediction_history.csv")


# ============================================================
# VALIDATION
# ============================================================

def validate_prediction_input(data):
    if not isinstance(data, dict):
        return False, "Payload must be a valid JSON object."

    required_fields = [
        "hour",
        "day_of_week",
        "priority",
        "unit_type",
        "als_unit",
        "zipcode",
    ]

    for field in required_fields:
        if field not in data or data[field] is None or data[field] == "":
            return False, f"Missing required field: '{field}'"

    try:
        hour = int(data["hour"])
        if hour not in VALID_HOURS:
            return False, "Hour must be an integer between 0 and 23."
    except (ValueError, TypeError):
        return False, "Hour must be a valid integer."

    day = str(data["day_of_week"]).capitalize()
    if day not in VALID_DAYS_OF_WEEK:
        return False, "Invalid day_of_week."

    priority = str(data["priority"]).upper()
    if priority not in VALID_PRIORITIES:
        return False, f"Priority must be one of {VALID_PRIORITIES}"

    unit_type = str(data["unit_type"]).upper()
    if unit_type not in ["MEDIC", "PRIVATE"]:
        return False, "Unit type must be one of ['MEDIC', 'PRIVATE']"

    if not isinstance(data["als_unit"], bool):
        return False, "als_unit must be true or false."

    zipcode = str(data["zipcode"]).strip()
    if not zipcode:
        return False, "ZIP code cannot be empty."

    return True, None


# ============================================================
# HISTORICAL DATA (MEMORY EFFICIENT)
# ============================================================

def _load_historical_data():
    """
    Memory-optimized loader for Render free tier (512 MB limit).
    Calculates summary lookup statistics instead of heavy expanding windows.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Historical dataset not found: {DATA_PATH}")

    use_cols = [
        "received_dttm",
        "response_dttm",
        "station_area",
        "neighborhoods_analysis_boundaries",
        "zipcode_of_incident",
    ]

    df = pd.read_csv(DATA_PATH, usecols=use_cols, low_memory=False)

    df["received_dttm"] = pd.to_datetime(df["received_dttm"], errors="coerce")
    df["response_dttm"] = pd.to_datetime(df["response_dttm"], errors="coerce")

    df["response_time_minutes"] = (
        df["response_dttm"] - df["received_dttm"]
    ).dt.total_seconds() / 60.0

    df = df[
        df["received_dttm"].notna()
        & df["response_time_minutes"].notna()
        & (df["response_time_minutes"] > 0)
    ].copy()

    df["hour"] = df["received_dttm"].dt.hour

    categorical_cols = [
        "station_area",
        "neighborhoods_analysis_boundaries",
        "zipcode_of_incident",
    ]
    for col in categorical_cols:
        df[col] = df[col].astype(str).fillna("UNKNOWN").str.strip()

    return df


_HISTORICAL_DATA = None


def _get_historical_data():
    global _HISTORICAL_DATA
    if _HISTORICAL_DATA is None:
        print("Loading historical prediction data...")
        _HISTORICAL_DATA = _load_historical_data()
        print(f"Historical prediction data loaded: {len(_HISTORICAL_DATA):,} records.")
    return _HISTORICAL_DATA


# ============================================================
# FEATURE PREPROCESSING
# ============================================================

def preprocess_features(raw_data):
    hour = int(raw_data["hour"])
    day_name = str(raw_data["day_of_week"]).capitalize()
    day_mapping = {day: index for index, day in enumerate(VALID_DAYS_OF_WEEK)}
    day_of_week = day_mapping[day_name]
    is_weekend = int(day_name in ["Saturday", "Sunday"])

    priority = str(raw_data["priority"]).upper()
    unit_type = str(raw_data["unit_type"]).upper()
    als_unit = int(raw_data["als_unit"])
    zipcode = str(raw_data["zipcode"]).strip()

    station_area = str(raw_data.get("station_area", "01")).strip()
    neighborhood = str(raw_data.get("neighborhood", "Unknown")).strip()

    original_priority = str(raw_data.get("original_priority", priority)).upper()
    battalion = str(raw_data.get("battalion", "B02")).strip()
    fire_prevention_district = str(raw_data.get("fire_prevention_district", "2")).strip()
    supervisor_district = str(raw_data.get("supervisor_district", "6")).strip()

    # Time features
    minutes_since_midnight = hour * 60
    time_sin = np.sin(2 * np.pi * minutes_since_midnight / 1440)
    time_cos = np.cos(2 * np.pi * minutes_since_midnight / 1440)

    now = pd.Timestamp.now()
    month = now.month
    day_of_month = now.day

    # Historical features lookup
    historical = _get_historical_data()
    global_mean = historical["response_time_minutes"].mean()

    station_rows = historical[historical["station_area"] == station_area]
    neighborhood_rows = historical[
        historical["neighborhoods_analysis_boundaries"] == neighborhood
    ]
    zip_rows = historical[historical["zipcode_of_incident"] == zipcode]
    hour_rows = historical[historical["hour"] == hour]

    station_hour_rows = historical[
        (historical["station_area"] == station_area) & (historical["hour"] == hour)
    ]
    neighborhood_hour_rows = historical[
        (historical["neighborhoods_analysis_boundaries"] == neighborhood)
        & (historical["hour"] == hour)
    ]

    def mean_or_global(rows):
        if len(rows) == 0:
            return global_mean
        val = rows["response_time_minutes"].mean()
        return val if pd.notna(val) else global_mean

    features = {
        "priority": [priority],
        "original_priority": [original_priority],
        "unit_type": [unit_type],
        "zipcode_of_incident": [zipcode],
        "station_area": [station_area],
        "neighborhoods_analysis_boundaries": [neighborhood],
        "battalion": [battalion],
        "fire_prevention_district": [fire_prevention_district],
        "supervisor_district": [supervisor_district],
        "hour": [hour],
        "day_of_week": [day_of_week],
        "month": [month],
        "day_of_month": [day_of_month],
        "is_weekend": [is_weekend],
        "time_sin": [time_sin],
        "time_cos": [time_cos],
        "als_unit": [als_unit],
        "historical_global_mean": [global_mean],
        "historical_station_mean": [mean_or_global(station_rows)],
        "historical_neighborhood_mean": [mean_or_global(neighborhood_rows)],
        "historical_zip_mean": [mean_or_global(zip_rows)],
        "historical_hour_mean": [mean_or_global(hour_rows)],
        "historical_station_hour_mean": [mean_or_global(station_hour_rows)],
        "historical_neighborhood_hour_mean": [mean_or_global(neighborhood_hour_rows)],
        "historical_station_call_count": [len(station_rows)],
        "historical_neighborhood_call_count": [len(neighborhood_rows)],
        "historical_hour_call_count": [len(hour_rows)],
    }

    return pd.DataFrame(features)