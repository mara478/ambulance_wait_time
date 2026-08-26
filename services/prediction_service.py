
"""
Machine-learning prediction service.

This module is responsible for:

1. Loading the trained ambulance response-time model.
2. Preparing prediction input.
3. Generating a response-time prediction.

The final trained model will be stored at:

model/ambulance_response_model.joblib
"""

import os
import joblib


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    "model",
    "ambulance_response_model.joblib"
)


# ============================================================
# MODEL SERVICE
# ============================================================

class PredictionService:
    """
    Handles loading and using the ambulance response-time model.
    """

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.load_error = None

        self._load_model()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    def _load_model(self):
        """
        Load the trained model from disk.

        The model does not exist yet during development,
        so the application should not crash if the file
        is missing.
        """

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
            self.load_error = str(error)

    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    def is_ready(self):
        """
        Return True if the model was successfully loaded.
        """

        return self.model is not None

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    def predict(self, features):
        """
        Generate an ambulance response-time prediction.

        Parameters
        ----------
        features:
            Data prepared in the exact format expected
            by the trained model.

        Returns
        -------
        float
            Predicted response time in minutes.

        Raises
        ------
        RuntimeError
            If the model has not been loaded.
        """

        if not self.is_ready():
            raise RuntimeError(
                "The ambulance response-time model is not available."
            )

        try:
            prediction = self.model.predict(features)

            # The model returns an array-like object.
            # Convert the first prediction to a normal float.
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

