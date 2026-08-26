import csv
from collections import defaultdict
from datetime import datetime

INPUT_FILE = "data/ambulance_data.csv"
OUTPUT_FILE = "data/ambulance_ml_data.csv"


def parse_datetime(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


incidents = defaultdict(list)


# Read the original CSV
with open(INPUT_FILE, "r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        incidents[row["incident_number"]].append(row)


ml_rows = []


# Create one row per incident
for incident_number, rows in incidents.items():

    # Sort units by response time
    rows_with_response = [
        row for row in rows
        if parse_datetime(row["response_dttm"]) is not None
    ]

    if not rows_with_response:
        continue

    rows_with_response.sort(
        key=lambda row: parse_datetime(row["response_dttm"])
    )

    # Use the first responding unit
    row = rows_with_response[0]

    received = parse_datetime(row["received_dttm"])
    on_scene = parse_datetime(row["on_scene_dttm"])

    # We need both timestamps to calculate response time
    if received is None or on_scene is None:
        continue

    response_minutes = (
        on_scene - received
    ).total_seconds() / 60

    # Ignore impossible values
    if response_minutes < 0 or response_minutes > 120:
        continue

    call_date = parse_datetime(row["call_date"])

    if call_date is None:
        continue

    ml_rows.append({
        "incident_number": incident_number,
        "call_date": call_date.date().isoformat(),
        "day_of_week": call_date.strftime("%A"),
        "hour": call_date.hour,
        "call_type": row["call_type"],
        "call_type_group": row["call_type_group"],
        "priority": row["priority"],
        "original_priority": row["original_priority"],
        "als_unit": row["als_unit"],
        "zipcode": row["zipcode_of_incident"],
        "station_area": row["station_area"],
        "neighborhood": row["neighborhoods_analysis_boundaries"],
        "response_time_minutes": round(response_minutes, 2)
    })


# Save the ML dataset
fieldnames = [
    "incident_number",
    "call_date",
    "day_of_week",
    "hour",
    "call_type",
    "call_type_group",
    "priority",
    "original_priority",
    "als_unit",
    "zipcode",
    "station_area",
    "neighborhood",
    "response_time_minutes"
]


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(ml_rows)


print("Finished!")
print("Original rows:", sum(len(rows) for rows in incidents.values()))
print("Unique incidents:", len(incidents))
print("ML rows created:", len(ml_rows))
print("Saved to:", OUTPUT_FILE)