"""
Preprocessing and feature engineering utilities for
ambulance response-time prediction.

This module validates the data received from the frontend
and converts it into the exact feature format expected
by the trained machine-learning model.
"""

import pandas as pd


# ============================================================
# VALIDATION CONSTANTS
# ============================================================

VALID_PRIORITIES = ["1", "2", "3", "E", "I"]

VALID_UNIT_TYPES = [
    "MEDIC",
    "PRIVATE"
]

VALID_HOURS = list(range(24))

VALID_DAYS_OF_WEEK = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_prediction_input(data):
    """
    Validate the JSON payload received from the frontend.
    """

    if not isinstance(data, dict):
        return False, "Payload must be a valid JSON object."

    required_fields = [
        "hour",
        "day_of_week",
        "priority",
        "unit_type",
        "als_unit",
        "zipcode"
    ]

    for field in required_fields:

        if (
            field not in data
            or data[field] is None
            or data[field] == ""
        ):
            return False, (
                f"Missing required field: '{field}'"
            )

    # --------------------------------------------------------
    # Hour
    # --------------------------------------------------------

    try:

        hour = int(data["hour"])

        if hour not in VALID_HOURS:
            return False, (
                "Hour must be an integer between 0 and 23."
            )

    except (ValueError, TypeError):

        return False, (
            "Hour must be a valid integer."
        )


    # --------------------------------------------------------
    # Day
    # --------------------------------------------------------

    day = str(
        data["day_of_week"]
    ).capitalize()

    if day not in VALID_DAYS_OF_WEEK:

        return False, (
            "Invalid day_of_week."
        )


    # --------------------------------------------------------
    # Priority
    # --------------------------------------------------------

    priority = str(
        data["priority"]
    ).upper()

    if priority not in VALID_PRIORITIES:

        return False, (
            "Priority must be one of "
            f"{VALID_PRIORITIES}"
        )


    # --------------------------------------------------------
    # Unit type
    # --------------------------------------------------------

    unit_type = str(
        data["unit_type"]
    ).upper()

    if unit_type not in VALID_UNIT_TYPES:

        return False, (
            "Unit type must be one of "
            f"{VALID_UNIT_TYPES}"
        )


    # --------------------------------------------------------
    # ALS
    # --------------------------------------------------------

    if not isinstance(
        data["als_unit"],
        bool
    ):

        return False, (
            "als_unit must be true or false."
        )


    # --------------------------------------------------------
    # ZIP CODE
    # --------------------------------------------------------

    zipcode = str(
        data["zipcode"]
    ).strip()

    if not zipcode:

        return False, (
            "ZIP code cannot be empty."
        )


    return True, None


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def preprocess_features(raw_data):
    """
    Convert frontend data into the exact feature structure
    expected by the trained ML model.
    """

    # --------------------------------------------------------
    # Temporal features
    # --------------------------------------------------------

    hour = int(
        raw_data["hour"]
    )

    day_name = str(
        raw_data["day_of_week"]
    ).capitalize()

    day_mapping = {
        day: index
        for index, day
        in enumerate(VALID_DAYS_OF_WEEK)
    }

    day_of_week = day_mapping[
        day_name
    ]

    is_weekend = int(
        day_name in [
            "Saturday",
            "Sunday"
        ]
    )


    # --------------------------------------------------------
    # Emergency characteristics
    # --------------------------------------------------------

    priority = str(
        raw_data["priority"]
    ).upper()

    unit_type = str(
        raw_data["unit_type"]
    ).upper()

    als_unit = int(
        raw_data["als_unit"]
    )


    # --------------------------------------------------------
    # Geographic features
    # --------------------------------------------------------

    zipcode = str(
        raw_data["zipcode"]
    ).strip()

    station_area = str(
        raw_data.get(
            "station_area",
            "01"
        )
    ).strip()

    neighborhood = str(
        raw_data.get(
            "neighborhood",
            "Unknown"
        )
    ).strip()


    # --------------------------------------------------------
    # Additional model features
    #
    # These are currently supplied by the user/interface
    # or assigned safe defaults.
    # --------------------------------------------------------

    call_type_group = str(
        raw_data.get(
            "call_type_group",
            "Potentially Life-Threatening"
        )
    ).strip()

    battalion = str(
        raw_data.get(
            "battalion",
            "B02"
        )
    ).strip()

    fire_prevention_district = str(
        raw_data.get(
            "fire_prevention_district",
            "2"
        )
    ).strip()

    supervisor_district = str(
        raw_data.get(
            "supervisor_district",
            "6"
        )
    ).strip()


    # --------------------------------------------------------
    # Create model feature row
    # --------------------------------------------------------

    features = {

        "hour": [hour],

        "day_of_week": [day_of_week],

        "is_weekend": [is_weekend],

        "priority": [priority],

        "als_unit": [als_unit],

        "unit_type": [unit_type],

        "zipcode_of_incident": [zipcode],

        "station_area": [station_area],

        "neighborhoods_analysis_boundaries": [
            neighborhood
        ],

        "call_type_group": [
            call_type_group
        ],

        "battalion": [
            battalion
        ],

        "fire_prevention_district": [
            fire_prevention_district
        ],

        "supervisor_district": [
            supervisor_district
        ]
    }


    return pd.DataFrame(
        features
    )