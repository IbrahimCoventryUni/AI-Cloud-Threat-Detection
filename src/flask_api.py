from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import os
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Requests

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "../models/random_forest_unsw.pkl")
model = joblib.load(model_path)

# Debug: Print expected feature names from the model
try:
    correct_feature_names = model.feature_names_in_
    print("✅ Model was trained with these feature names:", correct_feature_names)
except AttributeError:
    print("⚠️ Warning: Model does not store feature names.")

# Define API Key (for security)
API_KEY = "9003121034"

# Define the selected features used during model training
SELECTED_FEATURES = [
    'sttl', 'ct_state_ttl', 'dttl', 'smeansz', 'ct_srv_dst',
    'ct_dst_sport_ltm', 'ct_srv_src', 'sbytes', 'service', 'swin',
    'ct_dst_src_ltm', 'ct_src_dport_ltm', 'sport', 'state', 'Ltime',
    'dwin', 'ct_src_ltm', 'Stime', 'dmeansz', 'dsport'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Authenticate API request
        api_key = request.headers.get("x-api-key")
        if api_key != API_KEY:
            return jsonify({"error": "Unauthorized. Invalid API Key"}), 403

        # Get JSON data from request
        data = request.json
        input_features = np.array(data["features"]).reshape(1, -1)

        # Ensure input features match expected number
        if len(input_features[0]) != len(SELECTED_FEATURES):
            return jsonify({"error": f"Expected {len(SELECTED_FEATURES)} features, but got {len(input_features[0])}"}), 400

        # Create a DataFrame with the correct column names
        features_df = pd.DataFrame(input_features, columns=SELECTED_FEATURES)

        # Debugging Output
        print("✅ DataFrame for prediction:\n", features_df.head())

        # Make prediction
        prediction = model.predict(features_df)
        result = "BENIGN" if prediction[0] == 0 else "THREAT"

        return jsonify({"prediction": result})

    except Exception as e:
        print("🔴 Error occurred:", str(e))  # Debugging output
        return jsonify({"error": str(e)}), 400

# ✅ Fix for favicon.ico requests (prevent 404 errors)
@app.route('/favicon.ico')
def favicon():
    return "", 204  # Respond with an empty favicon to prevent errors

# ✅ Fix for incorrect HTTP method usage
@app.route('/predict', methods=['GET'])
def predict_get():
    return jsonify({"error": "Hi Antal, Use a POST request with valid JSON input."}), 405

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
