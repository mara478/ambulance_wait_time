"""
Machine-learning prediction service.

Loads the trained ambulance response-time model and
generates response-time predictions.
"""

import os
import joblib


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
            prediction = self.model.predict(features)

            predicted_minutes = float(prediction[0])

            return predicted_minutes

        except Exception as error:
            raise RuntimeError(
                f"Prediction failed: {error}"
            ) from error


# ============================================================
# SINGLE MODEL INSTANCE
# ============================================================

prediction_service = PredictionService()