import csv

FILE = "data/ambulance_data.csv"

incidents = set()
rows = 0

with open(FILE, "r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        rows += 1
        incidents.add(row["incident_number"])

print("Total rows:", rows)
print("Unique incidents:", len(incidents))