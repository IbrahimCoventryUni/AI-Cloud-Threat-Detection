from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import os
import pandas as pd
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Requests

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "../models/ai_threat_model.pkl")
model = joblib.load(model_path)

# Define API Key (for security)
API_KEY = "9003121034"  

@app.route('/predict', methods=['POST'])
def predict():
    try:
        api_key = request.headers.get("x-api-key")
        if api_key != API_KEY:
            return jsonify({"error": "Unauthorized. Invalid API Key"}), 403
        
        # Debug: Print incoming request
        print("Received Request:", request.json)  

        # Ensure JSON data is received
        if not request.json or "features" not in request.json:
            return jsonify({"error": "Invalid JSON format. 'features' key missing"}), 400

        features = np.array(request.json["features"]).reshape(1, -1)

        # Convert NumPy array to DataFrame with column names
        column_names = ["feature_" + str(i) for i in range(features.shape[1])]
        features_df = pd.DataFrame(features, columns=column_names)

        # Debug: Print feature DataFrame
        print("Feature DataFrame:\n", features_df)

        # Make prediction
        prediction = model.predict(features_df)
        result = "BENIGN" if prediction[0] == "BENIGN" else "THREAT"

        return jsonify({"prediction": result})

    except Exception as e:
        print("Error:", str(e))  # Debugging output
        return jsonify({"error": str(e)}), 400

    
# ✅ Fix for favicon.ico requests (prevent 404 errors)
@app.route('/favicon.ico')
def favicon():
    return "", 204  # Respond with an empty favicon to prevent errors

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
