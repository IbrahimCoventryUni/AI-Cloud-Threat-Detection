from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import os
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Requests

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "../models/neural_network_unsw.pkl")
model = joblib.load(model_path)

# Debug: Print expected feature names from the model
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
        # Authenticate API request
        api_key = request.headers.get("x-api-key")
        if api_key != API_KEY:
            return jsonify({"error": "Unauthorized. Invalid API Key"}), 403
        
        # Get JSON data from request
        data = request.json
        input_features = np.array(data["features"]).reshape(1, -1)

        # Expected feature names from the trained model (46 features)
        correct_feature_names = [
            'sport', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
            'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin',
            'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit',
            'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
            'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
            'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
            'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat'
        ]

        # Ensure input features match expected order and number
        if len(input_features[0]) != len(correct_feature_names):
            return jsonify({"error": f"Expected {len(correct_feature_names)} features, but got {len(input_features[0])}"}), 400

        # Convert input features into DataFrame
        features_df = pd.DataFrame(input_features, columns=correct_feature_names)

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
    return jsonify({"error": "Use a POST request with valid JSON input."}), 405

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
