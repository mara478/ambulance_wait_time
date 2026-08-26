"""
Main application backend for the AI Ambulance Response & First-Aid Assistant.

This module initializes the Flask application, serves the frontend interface, 
exposes prediction and health-check REST API endpoints, and coordinates 
feature preprocessing with the machine learning model.
"""

import os
from flask import Flask, render_template, request, jsonify

from services.prediction_service import prediction_service
from utils.preprocessing import validate_prediction_input, preprocess_features


# ============================================================
# APPLICATION INITIALIZATION
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)


# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def index():
    """
    Renders the main web interface.
    """
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """
    Health-check endpoint for deployment monitoring (Render) and status checks.
    """
    model_ready = prediction_service.is_ready()
    
    return jsonify({
        "status": "ok",
        "model_loaded": model_ready,
        "model_error": prediction_service.load_error if not model_ready else None
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles prediction requests from the frontend form/API.
    
    Expects a JSON payload containing emergency call features, validates the inputs,
    preprocesses them into model features, and returns the predicted response time in minutes.
    """
    # 1. Parse JSON Request
    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "error": "Invalid payload. Expected JSON body."
        }), 400

    # 2. Input Validation
    is_valid, error_msg = validate_prediction_input(data)
    if not is_valid:
        return jsonify({
            "error": error_msg
        }), 400

    # 3. Verify Model Availability
    if not prediction_service.is_ready():
        return jsonify({
            "error": "Prediction service unavailable. Model not loaded.",
            "details": prediction_service.load_error
        }), 503

    # 4. Feature Preprocessing & Prediction Pipeline
    try:
        # Preprocess features into a model-compatible DataFrame
        features_df = preprocess_features(data)

        # Generate prediction (minutes)
        predicted_minutes = prediction_service.predict(features_df)

        return jsonify({
            "status": "success",
            "predicted_response_time_minutes": round(predicted_minutes, 2)
        }), 200

    except Exception as e:
        return jsonify({
            "error": "An error occurred during prediction processing.",
            "details": str(e)
        }), 500


# ============================================================
# ENTRY POINT (LOCAL DEVELOPMENT)
# ============================================================

if __name__ == "__main__":
    # Retrieve port from environment for deployment compatibility
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)