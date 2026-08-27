"""
Preprocessing utilities using precomputed summary statistics.
"""

import os
import json
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
STATS_PATH = os.path.join(PROJECT_DIR, "data", "summary_stats.json")

_SUMMARY_STATS = None


def _get_summary_stats():
    global _SUMMARY_STATS
    if _SUMMARY_STATS is None:
        if not os.path.exists(STATS_PATH):
            raise FileNotFoundError(f"Precomputed stats file missing: {STATS_PATH}")
        with open(STATS_PATH, "r") as f:
            _SUMMARY_STATS = json.load(f)
    return _SUMMARY_STATS


def validate_prediction_input(data):
    if not isinstance(data, dict):
        return False, "Payload must be a valid JSON object."

    required_fields = ["hour", "day_of_week", "priority", "unit_type", "als_unit", "zipcode"]
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

    minutes_since_midnight = hour * 60
    time_sin = np.sin(2 * np.pi * minutes_since_midnight / 1440)
    time_cos = np.cos(2 * np.pi * minutes_since_midnight / 1440)

    now = pd.Timestamp.now()
    month = now.month
    day_of_month = now.day

    stats = _get_summary_stats()
    global_mean = stats["global_mean"]

    station_mean = stats["station_mean"].get(station_area, global_mean)
    neighborhood_mean = stats["neighborhood_mean"].get(neighborhood, global_mean)
    zip_mean = stats["zip_mean"].get(zipcode, global_mean)
    hour_mean = stats["hour_mean"].get(str(hour), global_mean)

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
        "historical_station_mean": [station_mean],
        "historical_neighborhood_mean": [neighborhood_mean],
        "historical_zip_mean": [zip_mean],
        "historical_hour_mean": [hour_mean],
        "historical_station_hour_mean": [station_mean],
        "historical_neighborhood_hour_mean": [neighborhood_mean],
        "historical_station_call_count": [100],
        "historical_neighborhood_call_count": [100],
        "historical_hour_call_count": [100],
    }

    return pd.DataFrame(features)
