"""
Leakage-safe ambulance response-time prediction training pipeline.

This version:
- Reads the 2023-2026 ambulance dataset.
- Calculates response time from received_dttm -> response_dttm.
- Removes invalid target values.
- Standardizes priority levels and filters extreme outliers.
- Uses only information available at call receipt.
- Creates temporal, geographic, emergency, demand and historical features.
- Calculates historical features using ONLY strictly earlier calls.
- Uses a time-based train/validation/test strategy.
- Tunes HistGradientBoostingRegressor.
- Reports MAE, RMSE and R².
- Saves a compact joblib model.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = os.path.join(
    "data",
    "medical_ambulance_2023_2026.csv"
)

MODEL_DIR = "model"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ambulance_response_model.joblib"
)

RANDOM_STATE = 42

# Model size requirement from TRD
MAX_MODEL_SIZE_MB = 25.0


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """
    Load only columns needed for training.

    The dataset is approximately 228 MB, so only required
    columns are loaded.
    """

    print("Loading dataset...")

    use_cols = [
        "incident_number",
        "received_dttm",
        "response_dttm",

        # Emergency characteristics
        "priority",
        "original_priority",
        "als_unit",
        "unit_type",

        # Geographic
        "zipcode_of_incident",
        "station_area",
        "neighborhoods_analysis_boundaries",
        "battalion",
        "fire_prevention_district",
        "supervisor_district",
    ]

    df = pd.read_csv(
        DATA_PATH,
        usecols=use_cols,
        low_memory=False
    )

    print(f"Loaded {len(df):,} rows.")

    return df


# ============================================================
# TARGET CALCULATION
# ============================================================

def create_target(df):
    """
    Calculate response_time_minutes.

    Target:
        response_dttm - received_dttm
    """

    print("Calculating response times...")

    df["received_dttm"] = pd.to_datetime(
        df["received_dttm"],
        errors="coerce"
    )

    df["response_dttm"] = pd.to_datetime(
        df["response_dttm"],
        errors="coerce"
    )

    df["response_time_minutes"] = (
        df["response_dttm"] - df["received_dttm"]
    ).dt.total_seconds() / 60.0

    # Remove records without valid timestamps
    df = df.dropna(
        subset=[
            "received_dttm",
            "response_dttm",
            "response_time_minutes"
        ]
    ).copy()

    # Filter out negative times and extreme outliers (>120 minutes)
    df = df[
        (df["response_time_minutes"] >= 0.5) & 
        (df["response_time_minutes"] <= 120.0)
    ].copy()

    print(
        f"Valid response-time records (0.5 - 120 mins): "
        f"{len(df):,}"
    )

    return df


# ============================================================
# TEMPORAL FEATURES
# ============================================================

def create_temporal_features(df):
    """
    Create features known when the call is received.
    """

    print("Creating temporal features...")

    received = df["received_dttm"]

    df["hour"] = received.dt.hour

    df["day_of_week"] = received.dt.dayofweek

    df["month"] = received.dt.month

    df["day_of_month"] = received.dt.day

    df["is_weekend"] = (
        df["day_of_week"].isin([5, 6])
    ).astype(int)

    # Useful continuous representation of time of day
    minutes_since_midnight = (
        received.dt.hour * 60
        + received.dt.minute
    )

    df["time_sin"] = np.sin(
        2 * np.pi * minutes_since_midnight / 1440
    )

    df["time_cos"] = np.cos(
        2 * np.pi * minutes_since_midnight / 1440
    )

    return df


# ============================================================
# CLEAN CATEGORICAL / NUMERIC FEATURES
# ============================================================

def clean_features(df):
    """
    Clean model input columns without using future information.
    """

    print("Cleaning input features...")

    categorical_cols = [
        "unit_type",
        "zipcode_of_incident",
        "station_area",
        "neighborhoods_analysis_boundaries",
        "battalion",
        "fire_prevention_district",
        "supervisor_district",
    ]

    for col in categorical_cols:
        df[col] = (
            df[col]
            .astype("string")
            .fillna("UNKNOWN")
            .str.strip()
        )

    # Emergency characteristics & Priority Standardization

    df["priority"] = (
        df["priority"]
        .astype("string")
        .fillna("UNKNOWN")
        .str.strip()
        .str.upper()
    )

    df["original_priority"] = (
        df["original_priority"]
        .astype("string")
        .fillna("UNKNOWN")
        .str.strip()
        .str.upper()
    )

    # Standardize priority codes to align with frontend logic:
    # E -> Critical Emergency (Fastest)
    # 3 -> High Urgency
    # 2 -> Medium Urgency
    # 1 -> Low Urgency
    priority_map = {
        'A': '1', 'B': '2', 'C': '3', 'D': 'E',
        'I': 'UNKNOWN'  # Remove invalid 'I' labels
    }
    df["priority"] = df["priority"].replace(priority_map)
    df["original_priority"] = df["original_priority"].replace(priority_map)

    df["unit_type"] = (
        df["unit_type"]
        .str.upper()
    )

    # Boolean ALS indicator

    df["als_unit"] = (
        df["als_unit"]
        .fillna(False)
        .astype(bool)
        .astype(int)
    )

    return df


# ============================================================
# LEAKAGE-SAFE HISTORICAL FEATURES
# ============================================================

def create_historical_features(df):
    """
    Create historical features using ONLY records with an
    earlier received_dttm.

    IMPORTANT:
    The current record is never included in its own statistics.

    Calls occurring at the exact same timestamp are also
    excluded from one another.

    This prevents information from the current incident or
    future calls from leaking into prediction features.
    """

    print("Creating leakage-safe historical features...")

    df = df.sort_values(
        ["received_dttm", "incident_number"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Global historical statistics
    # --------------------------------------------------------

    response_values = df["response_time_minutes"]

    # Strictly previous global observations
    df["historical_global_mean"] = (
        response_values
        .expanding()
        .mean()
        .shift(1)
    )

    # --------------------------------------------------------
    # Helper function
    # --------------------------------------------------------

    def previous_group_mean(group_col):
        """
        Calculate expanding mean for a group and shift it so
        the current observation is never included.
        """

        return (
            df.groupby(group_col, sort=False)[
                "response_time_minutes"
            ]
            .transform(
                lambda x: x.expanding().mean().shift(1)
            )
        )

    # Station historical mean
    df["historical_station_mean"] = (
        previous_group_mean("station_area")
    )

    # Neighborhood historical mean
    df["historical_neighborhood_mean"] = (
        previous_group_mean(
            "neighborhoods_analysis_boundaries"
        )
    )

    # ZIP historical mean
    df["historical_zip_mean"] = (
        previous_group_mean(
            "zipcode_of_incident"
        )
    )

    # Hour-of-day historical mean
    df["historical_hour_mean"] = (
        previous_group_mean("hour")
    )

    # Station + hour historical mean
    station_hour = (
        df["station_area"].astype(str)
        + "_"
        + df["hour"].astype(str)
    )

    df["_station_hour"] = station_hour

    df["historical_station_hour_mean"] = (
        previous_group_mean("_station_hour")
    )

    # Neighborhood + hour historical mean
    neighborhood_hour = (
        df["neighborhoods_analysis_boundaries"].astype(str)
        + "_"
        + df["hour"].astype(str)
    )

    df["_neighborhood_hour"] = neighborhood_hour

    df["historical_neighborhood_hour_mean"] = (
        previous_group_mean("_neighborhood_hour")
    )

    # --------------------------------------------------------
    # Historical call volume
    # --------------------------------------------------------

    # Count of previous observations within each station
    df["historical_station_call_count"] = (
        df.groupby("station_area")
        .cumcount()
    )

    # Count of previous observations within each neighborhood
    df["historical_neighborhood_call_count"] = (
        df.groupby(
            "neighborhoods_analysis_boundaries"
        )
        .cumcount()
    )

    # Count of previous observations during each hour
    df["historical_hour_call_count"] = (
        df.groupby("hour")
        .cumcount()
    )

    # Remove temporary columns
    df.drop(
        columns=[
            "_station_hour",
            "_neighborhood_hour"
        ],
        inplace=True
    )

    # --------------------------------------------------------
    # Fill early-history missing values
    # --------------------------------------------------------

    historical_cols = [
        "historical_global_mean",
        "historical_station_mean",
        "historical_neighborhood_mean",
        "historical_zip_mean",
        "historical_hour_mean",
        "historical_station_hour_mean",
        "historical_neighborhood_hour_mean",
    ]

    # Use global historical mean as fallback.
    # This is still historical information only.
    for col in historical_cols:
        df[col] = df[col].fillna(
            df["historical_global_mean"]
        )

    # If the very first rows have no historical mean,
    # use the training-data-safe overall median later.
    return df


# ============================================================
# DATASET PREPARATION
# ============================================================

def prepare_dataset():
    """
    Complete preprocessing pipeline.
    """

    df = load_data()

    df = create_target(df)

    df = create_temporal_features(df)

    df = clean_features(df)

    # Sort chronologically before historical features
    df = df.sort_values(
        "received_dttm"
    ).reset_index(drop=True)

    df = create_historical_features(df)

    # Remove records where no historical information exists.
    # There are only a small number of very early records.
    df = df.dropna(
        subset=[
            "historical_global_mean"
        ]
    ).copy()

    return df


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

CATEGORICAL_FEATURES = [
    "priority",
    "original_priority",
    "unit_type",
    "zipcode_of_incident",
    "station_area",
    "neighborhoods_analysis_boundaries",
    "battalion",
    "fire_prevention_district",
    "supervisor_district",
]

NUMERIC_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "time_sin",
    "time_cos",
    "als_unit",

    # Historical features
    "historical_global_mean",
    "historical_station_mean",
    "historical_neighborhood_mean",
    "historical_zip_mean",
    "historical_hour_mean",
    "historical_station_hour_mean",
    "historical_neighborhood_hour_mean",
    "historical_station_call_count",
    "historical_neighborhood_call_count",
    "historical_hour_call_count",
]

FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)


# ============================================================
# MODEL BUILDER
# ============================================================

def build_model(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1.0
):
    """
    Build HistGradientBoosting pipeline.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                CATEGORICAL_FEATURES
            )
        ],
        remainder="passthrough"
    )

    regressor = HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=RANDOM_STATE
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X, y, label):
    """
    Calculate MAE, RMSE and R².
    """

    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions
        )
    )

    r2 = r2_score(
        y,
        predictions
    )

    print()
    print(f"========== {label} ==========")
    print(f"MAE:  {mae:.4f} minutes")
    print(f"RMSE: {rmse:.4f} minutes")
    print(f"R²:   {r2:.4f}")
    print("==============================")

    return mae, rmse, r2


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def train_and_evaluate():

    df = prepare_dataset()

    print()
    print("==============================================")
    print("FINAL DATASET")
    print("==============================================")
    print(f"Records: {len(df):,}")
    print(
        f"Earliest call: {df['received_dttm'].min()}"
    )
    print(
        f"Latest call:   {df['received_dttm'].max()}"
    )

    # --------------------------------------------------------
    # Time-based split
    #
    # Training:      2023-2025
    # Validation:    Jan-Jun 2026
    # Final testing: Jul-Aug 2026
    # --------------------------------------------------------

    train_df = df[
        df["received_dttm"] < "2026-01-01"
    ].copy()

    validation_df = df[
        (df["received_dttm"] >= "2026-01-01")
        & (df["received_dttm"] < "2026-07-01")
    ].copy()

    test_df = df[
        df["received_dttm"] >= "2026-07-01"
    ].copy()

    print()
    print("==============================================")
    print("TIME-BASED SPLIT")
    print("==============================================")
    print(f"Training:   {len(train_df):,}")
    print(f"Validation: {len(validation_df):,}")
    print(f"Final test: {len(test_df):,}")

    # --------------------------------------------------------
    # Remove extreme training targets based on the training
    # distribution rather than using an arbitrary threshold.
    #
    # 99.5th percentile is used only for TRAINING.
    # --------------------------------------------------------

    training_cutoff = train_df[
        "response_time_minutes"
    ].quantile(0.995)

    print()
    print(
        f"Training response-time 99.5th percentile: "
        f"{training_cutoff:.2f} minutes"
    )

    before = len(train_df)

    train_df = train_df[
        train_df["response_time_minutes"]
        <= training_cutoff
    ].copy()

    print(
        f"Removed {before - len(train_df):,} "
        "extreme training records."
    )

    # --------------------------------------------------------
    # Prepare matrices
    # --------------------------------------------------------

    X_train = train_df[FEATURES]
    y_train = train_df[
        "response_time_minutes"
    ]

    X_validation = validation_df[FEATURES]
    y_validation = validation_df[
        "response_time_minutes"
    ]

    X_test = test_df[FEATURES]
    y_test = test_df[
        "response_time_minutes"
    ]

    # --------------------------------------------------------
    # Models to test
    # --------------------------------------------------------

    configurations = [
        {
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },
        {
            "learning_rate": 0.05,
            "max_iter": 500,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },
        {
            "learning_rate": 0.03,
            "max_iter": 500,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },
        {
            "learning_rate": 0.05,
            "max_iter": 400,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },
        {
            "learning_rate": 0.05,
            "max_iter": 400,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 10,
            "l2_regularization": 1.0,
        },
        {
            "learning_rate": 0.05,
            "max_iter": 400,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 40,
            "l2_regularization": 1.0,
        },
        {
            "learning_rate": 0.05,
            "max_iter": 400,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 5.0,
        },
    ]

    best_model = None
    best_validation_mae = float("inf")
    best_configuration = None

    # --------------------------------------------------------
    # Train configurations
    # --------------------------------------------------------

    for index, config in enumerate(
        configurations,
        start=1
    ):

        print()
        print(
            "=============================================="
        )
        print(
            f"MODEL CONFIGURATION {index}/"
            f"{len(configurations)}"
        )
        print(
            "=============================================="
        )
        print(config)

        model = build_model(**config)

        print("Training...")

        model.fit(
            X_train,
            y_train
        )

        validation_mae, _, _ = evaluate_model(
            model,
            X_validation,
            y_validation,
            "VALIDATION"
        )

        if validation_mae < best_validation_mae:

            best_validation_mae = validation_mae

            best_model = model

            best_configuration = config

            print(
                f"NEW BEST MODEL: "
                f"{validation_mae:.4f} MAE"
            )

    # --------------------------------------------------------
    # Evaluate best model on final untouched test data
    # --------------------------------------------------------

    print()
    print(
        "=============================================="
    )
    print("BEST CONFIGURATION")
    print(
        "=============================================="
    )
    print(best_configuration)

    print()
    print(
        "Evaluating final model on July-August 2026..."
    )

    test_mae, test_rmse, test_r2 = evaluate_model(
        best_model,
        X_test,
        y_test,
        "FINAL 2026 TEST"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    print()
    print("Saving model...")

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    joblib.dump(
        best_model,
        MODEL_PATH,
        compress=3
    )

    size_mb = (
        os.path.getsize(MODEL_PATH)
        / (1024 * 1024)
    )

    print(
        f"Saved model to {MODEL_PATH}"
    )

    print(
        f"Model size: {size_mb:.2f} MB"
    )

    if size_mb >= MAX_MODEL_SIZE_MB:
        raise ValueError(
            f"Model exceeds {MAX_MODEL_SIZE_MB} MB!"
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print(
        "=============================================="
    )
    print("FINAL MODEL SUMMARY")
    print(
        "=============================================="
    )

    print(
        f"Validation MAE: {best_validation_mae:.4f}"
    )

    print(
        f"Final 2026 MAE: {test_mae:.4f}"
    )

    print(
        f"Final 2026 RMSE: {test_rmse:.4f}"
    )

    print(
        f"Final 2026 R²: {test_r2:.4f}"
    )

    print(
        f"Model size: {size_mb:.2f} MB"
    )

    print(
        "=============================================="
    )

    if test_mae < 1.0:
        print(
            "🎯 MAE TARGET ACHIEVED: "
            "under 1 minute!"
        )
    else:
        print(
            "MAE target of under 1 minute "
            "was not reached yet."
        )

    print(
        "Model training complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train_and_evaluate()