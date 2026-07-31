"""
app.py
------
Task 3: API Development

A Flask REST API and Web Interface that loads the trained model (model.pkl),
accepts patient clinical parameters as JSON or via a UI form, and returns 
a heart-disease risk prediction.
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the trained artifact (model + scaler + expected feature order) once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
scaler = artifact["scaler"]
FEATURE_ORDER = artifact["feature_order"]

# HTML template embedded directly into the file so you don't need a separate templates/ folder
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Heart Disease Predictor</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; margin: 0; padding: 20px; }
        .container { max-width: 600px; background: white; margin: 20px auto; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h2 { margin-top: 0; color: #333; text-align: center; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .form-group { display: flex; flex-direction: column; }
        label { font-size: 0.9em; font-weight: bold; margin-bottom: 5px; color: #555; }
        input { padding: 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }
        button { grid-column: span 2; padding: 12px; background-color: #007bff; color: white; border: none; border-radius: 4px; font-size: 16px; font-weight: bold; cursor: pointer; margin-top: 10px; }
        button:hover { background-color: #0056b3; }
        #result { margin-top: 20px; padding: 15px; border-radius: 4px; display: none; font-weight: bold; text-align: center; }
        .detected { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .not-detected { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Heart Disease Predictor</h2>
        <form id="prediction-form" class="grid">
            <div class="form-group"><label>Age</label><input type="number" id="age" value="63" required></div>
            <div class="form-group"><label>Sex (1=Male, 0=Female)</label><input type="number" id="sex" value="1" min="0" max="1" required></div>
            <div class="form-group"><label>Chest Pain Type (0-3)</label><input type="number" id="cp" value="3" min="0" max="3" required></div>
            <div class="form-group"><label>Resting Blood Pressure</label><input type="number" id="trestbps" value="145" required></div>
            <div class="form-group"><label>Cholesterol (mg/dl)</label><input type="number" id="chol" value="233" required></div>
            <div class="form-group"><label>Fasting Blood Sugar > 120 (1/0)</label><input type="number" id="fbs" value="1" min="0" max="1" required></div>
            <div class="form-group"><label>Resting ECG (0-2)</label><input type="number" id="restecg" value="0" min="0" max="2" required></div>
            <div class="form-group"><label>Max Heart Rate (thalach)</label><input type="number" id="thalach" value="150" required></div>
            <div class="form-group"><label>Exercise Angina (1=Yes, 0=No)</label><input type="number" id="exang" value="0" min="0" max="1" required></div>
            <div class="form-group"><label>Oldpeak</label><input type="number" step="0.1" id="oldpeak" value="2.3" required></div>
            <div class="form-group"><label>ST Slope (0-2)</label><input type="number" id="slope" value="0" min="0" max="2" required></div>
            <div class="form-group"><label>Major Vessels (ca 0-4)</label><input type="number" id="ca" value="0" min="0" max="4" required></div>
            <div class="form-group" style="grid-column: span 2;"><label>Thalassemia (thal 0-3)</label><input type="number" id="thal" value="1" min="0" max="3" required></div>
            <button type="submit">Predict Risk</button>
        </form>

        <div id="result"></div>
    </div>

    <script>
        document.getElementById('prediction-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const fields = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"];
            const payload = {};
            fields.forEach(f => payload[f] = parseFloat(document.getElementById(f).value));

            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';

            if (data.prediction) {
                const isDetected = data.prediction.includes("Detected") && !data.prediction.includes("No");
                resultDiv.className = isDetected ? 'detected' : 'not-detected';
                resultDiv.innerHTML = `<strong>Result:</strong> ${data.prediction}<br><strong>Probability:</strong> ${(data.probability * 100).toFixed(2)}%`;
            } else {
                resultDiv.className = 'detected';
                resultDiv.innerText = `Error: ${data.error}`;
            }
        });
    </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    """Renders an interactive UI form directly in the browser."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api-info", methods=["GET"])
def api_info():
    """Returns JSON metadata about required API fields."""
    return jsonify({
        "message": "Heart Disease Prediction API is running.",
        "usage": "POST patient details as JSON to /predict",
        "required_fields": FEATURE_ORDER,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "No JSON payload received. Send patient details as JSON."}), 400

    # Validate that all required fields are present
    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        # Build DataFrame with explicit feature names to prevent StandardScaler warnings
        features_dict = {f: [float(data[f])] for f in FEATURE_ORDER}
        features_df = pd.DataFrame(features_dict)

        # Scale features and predict
        features_scaled = scaler.transform(features_df)

        pred = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0][1]  # probability of class "1"

        result = "Heart Disease Detected" if pred == 1 else "No Heart Disease Detected"

        return jsonify({
            "prediction": result,
            "probability": round(float(proba), 4)
        })

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input values: {str(e)}"}), 400


if __name__ == "__main__":
    # Render sets the PORT environment variable; default to 5000 for local runs
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)