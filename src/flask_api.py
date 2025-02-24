from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import os
import pandas as pd
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Requests

# Load the trained model
# model_path = os.path.join(os.path.dirname(__file__), "../models/ai_threat_model.pkl")

model_path = os.path.join(os.path.dirname(__file__), "../models/neural_network_unsw.pkl")
model = joblib.load(model_path)

# Debug: Print model feature names
try:
    correct_feature_names = model.feature_names_in_
    print("✅ Model was trained with these feature names:", correct_feature_names)
except AttributeError:
    print("⚠️ Warning: Model does not store feature names.")

# Define API Key (for security)
API_KEY = "9003121034"  

@app.route('/predict', methods=['POST'])
def predict():
    try:
        api_key = request.headers.get("x-api-key")
        if api_key != API_KEY:
            return jsonify({"error": "Unauthorized. Invalid API Key"}), 403
        
        # Get JSON data from request
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)

        # Use the correct feature names from the trained model
        correct_feature_names = model.feature_names_in_  # Extract original feature names
        features_df = pd.DataFrame(features, columns=correct_feature_names)  # Use correct names

        # Debug: Print DataFrame to check column names
        print("✅ DataFrame for prediction:\n", features_df.head())

        # Make prediction
        prediction = model.predict(features_df)
        result = "BENIGN" if prediction[0] == 0 else "THREAT"

        # prediction = model.predict(features_df)
        # result = "BENIGN" if prediction[0] == "BENIGN" else "THREAT"

        return jsonify({"prediction": result})

    except Exception as e:
        print("🔴 Error occurred:", str(e))  # Debugging output
        return jsonify({"error": str(e)}), 400

#     To overwrite the method not found page
@app.route('/predict', methods=['GET'])
def predict_get():
    return jsonify({"error": "Use a POST request with valid JSON input."}), 405


# ✅ Fix for favicon.ico requests (prevent 404 errors)
@app.route('/favicon.ico')
def favicon():
    return "", 204  # Respond with an empty favicon to prevent errors

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
