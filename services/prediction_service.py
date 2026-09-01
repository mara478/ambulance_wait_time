"""
Machine-learning prediction service.

Loads the trained ambulance response-time model and
generates response-time predictions.
"""

import os
import joblib
import pandas as pd


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Get the directory containing this file:
# bml-project/services/
SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the project root:
PROJECT_DIR = os.path.dirname(SERVICES_DIR)

# Build an absolute path to the model:
MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "model",
    "ambulance_response_model.joblib"
)


# ============================================================
# MODEL SERVICE
# ============================================================

class PredictionService:

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.load_error = None

        self._load_model()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    def _load_model(self):

        if not os.path.exists(self.model_path):
            self.load_error = (
                f"Model file not found: {self.model_path}"
            )
            return

        try:
            self.model = joblib.load(self.model_path)
            self.load_error = None

        except Exception as error:
            self.model = None
            self.load_error = (
                f"Failed to load model: {error}"
            )

    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    def is_ready(self):
        return self.model is not None

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    def predict(self, features):

        if not self.is_ready():
            raise RuntimeError(
                "The ambulance response-time model is not available."
            )

        try:
            # Convert dictionary input to DataFrame if needed
            if isinstance(features, dict):
                features_df = pd.DataFrame([features])
            elif isinstance(features, pd.DataFrame):
                features_df = features.copy()
            else:
                features_df = pd.DataFrame(features)

            # Ensure priority column values are clean strings
            if 'priority' in features_df.columns:
                features_df['priority'] = features_df['priority'].astype(str).str.strip().str.upper()

            # DEBUG LOGGING (Check VS Code Terminal Output)
            print("\n================ DEBUG PREDICTION ================")
            print(f"Incoming Priority: {features_df['priority'].iloc[0] if 'priority' in features_df.columns else 'N/A'}")
            print("Full Feature Data:")
            print(features_df.to_dict(orient='records')[0])

            # Run prediction
            prediction = self.model.predict(features_df)
            predicted_minutes = float(prediction[0])

            print(f"Predicted Response Time: {predicted_minutes:.2f} minutes")
            print("==================================================\n")

            return predicted_minutes

        except Exception as error:
            raise RuntimeError(
                f"Prediction failed: {error}"
            ) from error


# ============================================================
# SINGLE MODEL INSTANCE
# ============================================================

prediction_service = PredictionService()